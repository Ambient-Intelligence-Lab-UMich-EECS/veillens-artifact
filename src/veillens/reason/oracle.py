#!/usr/bin/python
#
# Structured-output oracle for privacy reasoning.
#
# Owns everything that is the same whichever model answers: prompt and schema selection (delegated
# to the privacy spec), concurrency, rate limiting, parsing, and the truncated-response debug dump.
# The provider-specific half -- how a request is built and issued -- lives behind a VLMClient in
# veillens.vlm. See docs/EXTENDING.md.
#
# Two clients, chosen per call by `role`:
#   role='reason'  the attacker inside the defense loop (rounds 1 and 2, the iteration gate)
#   role='eval'    the final measurement pass, whose numbers are compared across runs
# They default to the same OpenAI client, so a config that does not mention a local backend
# behaves exactly as it did before the split.

from __future__ import absolute_import, division, print_function

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import openai
from dotenv import load_dotenv

from veillens.vlm.base import VLMError

# Per-call failures go to the logger, not to print, so that a truncated response or a rate-limit
# retry lands in the same stream as the rest of the run's timeline rather than interleaved into
# whatever the caller happens to be printing.
logger = logging.getLogger(__name__)


class GeoLocationAPI:
    """
    Runs structured-output vision requests against one or two VLM backends.

    The schema, prompts, and Round 2 context formatting are all delegated to a
    PrivacyClassSpec, so the same class can drive geolocation removal or
    personal-information removal depending on configuration.
    """

    def __init__(self, model="gpt-5.1-2025-11-13", concurrent_requests=5, cache_path="", original_cache_dir="original_cache", eval_cache_dir="eval_cache", spec=None, reason_client=None, eval_client=None):
        load_dotenv()
        self.model = model
        self.concurrent_requests = concurrent_requests
        # Removed: cache_responses parameter and caching code

        # Rate limiting. This semaphore is the outer bound across both backends; each client also
        # holds its own limit, since a local server and a metered vendor API tolerate very
        # different concurrency.
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
        self.rate_limit_lock = asyncio.Lock()  # Thread-safe rate limiting

        if spec is None:
            from veillens.privacy.spec import GEOLOCATION_SPEC
            spec = GEOLOCATION_SPEC
        self.spec = spec

        if eval_client is None:
            from veillens.vlm.openai_client import OpenAIClient
            eval_client = OpenAIClient(model=model)
        self.eval_client = eval_client
        # Reasoning falls back to the measurement backend, which is what every pre-existing config
        # expects: one model, doing both jobs.
        self.reason_client = reason_client if reason_client is not None else eval_client

    def _client_for(self, role: str):
        if role == 'eval':
            return self.eval_client
        if role == 'reason':
            return self.reason_client
        raise ValueError(f"Unknown role {role!r}; expected 'reason' or 'eval'")

    async def _rate_limit(self):
        """Implement thread-safe rate limiting with atomic read-modify-write"""
        async with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)

            self.last_request_time = time.time()

    async def get_geolocation_prediction(self, image_path: str, schema_type: str = 'eval', predicted_location: str = None, region_name: str = None, cache_key_override: str = None, role: str = 'reason') -> Optional[Dict]:
        """
        Get geolocation prediction for a single image

        Args:
            image_path: Path to image file
            schema_type: Either 'original' (geo/non-geo pairs) or 'eval' (string arrays)
            predicted_location: For phase2 regional analysis, the predicted location from phase1
            region_name: For phase2 regional analysis, the region name (e.g., "top left")
            cache_key_override: Override cache key (used when image_path is temp file)
            role: Which backend answers -- 'reason' for the defense loop, 'eval' for the final
                measurement. Must be passed explicitly: schema_type does not imply it, because
                the compositional spec runs its final eval through the 'original' schema.

        Returns:
            Dict with prediction data or None if failed
        """
        client = self._client_for(role)
        start_time = time.time()

        is_round2 = schema_type == 'original' and predicted_location and region_name
        if schema_type == 'original':
            mode = 'r2' if is_round2 else 'r1'
            selected_schema = self.spec.get_schema(mode)
            schema_name = f"{self.spec.name}_original"
        else:
            selected_schema = self.spec.get_schema('eval')
            schema_name = f"{self.spec.name}_eval"

        # Removed: Cache checking code

        # Encode image BEFORE acquiring semaphore (file I/O doesn't need rate limiting)
        try:
            image_data_url = await asyncio.to_thread(client.encode_image, image_path)
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

        async with self.semaphore:
            await self._rate_limit()
            try:

                if schema_type == 'original':
                    if predicted_location and region_name:
                        selected_prompt = self.spec.format_round2_prompt(
                            predicted_location, region_name
                        )
                    else:
                        selected_prompt = self.spec.get_round1_prompt()
                else:
                    selected_prompt = self.spec.get_eval_prompt()

                # Names this call in any per-call record the backend keeps. Same stem the
                # truncated-response dump below uses, so the two line up.
                trace_stem = cache_key_override or Path(image_path).stem
                trace_label = f"{trace_stem}_{region_name}" if region_name else trace_stem

                # Make API request through the backend for this role
                api_start = time.time()
                data = await client.infer(
                    image_data_url=image_data_url,
                    prompt=selected_prompt,
                    schema=selected_schema,
                    schema_name=schema_name,
                    trace_label=trace_label,
                )
                api_time = time.time() - api_start

                # Track API timing

                # Parse the response
                obj = json.loads(data)

                # Add path identification
                obj['image_path'] = image_path
                # Removed: Cache saving code

                return obj

            except VLMError as e:
                # A backend-level failure worth shouting about: unlike a dropped request, it will
                # repeat for every image until the configuration is fixed.
                logger.error(f"[{role}] Backend error for {image_path}: {e}")
                return None

            except openai.APITimeoutError:
                logger.error(f"Timeout for image: {image_path}")
                return None

            except openai.APIError as e:
                logger.error(f"API error for {image_path}: {e}")
                return None

            except openai.OpenAIError as e:
                # Catches other OpenAI-specific errors (RateLimitError, AuthenticationError, etc.)
                logger.error(f"OpenAI error for {image_path}: {e}")
                return None

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error for {image_path}: {e}")
                logger.error(f"Response length: {len(data) if 'data' in locals() else 'N/A'} chars")

                # Save truncated response for debugging
                if 'data' in locals():
                    from datetime import datetime
                    from pathlib import Path as PathLib
                    debug_dir = PathLib("debug/truncated_responses")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_stem = cache_key_override if cache_key_override else PathLib(image_path).stem
                    if region_name:
                        debug_file = debug_dir / f"{timestamp}_{image_stem}_{region_name.replace(' ', '_')}_TRUNCATED.txt"
                    else:
                        debug_file = debug_dir / f"{timestamp}_{image_stem}_TRUNCATED.txt"

                    with open(debug_file, 'w') as f:
                        f.write(f"Error: {e}\n")
                        f.write(f"Error position: {e.pos}\n")
                        f.write(f"Response length: {len(data)}\n")
                        f.write(f"Image path: {image_path}\n")
                        f.write(f"Region: {region_name}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write("TRUNCATED RESPONSE:\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(data)

                    logger.error(f"Truncated response saved to: {debug_file}")

                return None

            except Exception as e:
                # Any other unexpected exception
                logger.error(f"Unexpected error for {image_path}: {e}")
                return None
