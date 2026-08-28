#!/usr/bin/env python3
"""
Build a geolocation manifest from a Cityscapes checkout.

This script generates JSON dataset files by discovering image/GPS pairs
from Cityscapes dataset structure. This separates data discovery from pipeline execution.

Usage:
python utils/create_dataset_json.py --cityscapes-root data --output data/dataset.json --cities aachen,cologne,erfurt,hanover,lindau,stuttgart,weimar,bochum,darmstadt,frankfurt,jena,monchengladbach,tubingen,zurich,bremen,dusseldorf,hamburg,krefeld,strasbourg,ulm --images-per-city 5
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to sys.path so we can import from core
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class CityscapesDiscovery:
    """
    Cityscapes dataset discovery utility.
    
    Replicates the original discovery logic from PipelineDataset to generate
    JSON files for the new JSON-based pipeline input system.
    """

    def __init__(self, cityscapes_root: str):
        self.cityscapes_root = Path(cityscapes_root)
        self.images_dir = self.cityscapes_root / "leftImg8bit" / "train"
        self.gps_dir = self.cityscapes_root / "vehicle" / "train"

        self.discovered_items = []
        self.items_by_city = defaultdict(list)

        # Validate directory structure
        self._validate_directory_structure()

    def _validate_directory_structure(self):
        """Validate that all required directories exist"""
        required_dirs = [self.images_dir, self.gps_dir]

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")

        logger.info(f"Dataset directory structure validated at: {self.cityscapes_root}")

    def discover_all_triplets(self) -> List[Dict]:
        """
        Discover all valid image/GPS pairs across all cities.

        Returns:
            List of dictionaries with discovery results
        """
        logger.info("Starting dataset discovery...")

        self.discovered_items = []
        self.items_by_city = defaultdict(list)

        # Find all cities
        cities = [d.name for d in self.images_dir.iterdir() if d.is_dir()]
        logger.info(f"Found cities: {cities}")

        total_found = 0
        total_valid = 0

        for city in cities:
            city_items = self._discover_city_triplets(city)
            valid_items = [item for item in city_items if self._is_valid_item(item)]

            self.discovered_items.extend(valid_items)
            self.items_by_city[city] = valid_items

            total_found += len(city_items)
            total_valid += len(valid_items)

            logger.info(f"City {city}: {len(city_items)} found, {len(valid_items)} valid")

        logger.info(f"Dataset discovery completed: {total_found} pairs found, {total_valid} valid")
        return self.discovered_items

    def _discover_city_triplets(self, city: str) -> List[Dict]:
        """Discover all image/GPS pairs for a specific city"""
        city_items = []

        city_images_dir = self.images_dir / city
        city_gps_dir = self.gps_dir / city

        if not all(d.exists() for d in [city_images_dir, city_gps_dir]):
            logger.warning(f"Incomplete directory structure for city: {city}")
            return city_items

        # Find all images in city
        image_files = sorted(list(city_images_dir.glob("*_leftImg8bit.png")))

        for image_path in image_files:
            base_name = image_path.stem.replace('_leftImg8bit', '')
            image_id = base_name

            gps_path = city_gps_dir / f"{base_name}_vehicle.json"

            # Extract GPS coordinates
            ground_truth_coords = self._extract_gps_coordinates(gps_path)

            item = {
                'image_id': image_id,
                'city': city,
                'image_path': str(image_path),
                # Path relative to the leftImg8bit split root, for the portable v2 manifest
                'relpath': str(image_path.relative_to(self.images_dir)),
                'ground_truth_coords': ground_truth_coords,
                'gps_path': str(gps_path)  # Keep for validation
            }

            city_items.append(item)

        return city_items

    def _extract_gps_coordinates(self, gps_path: Path) -> Optional[List[float]]:
        """Extract GPS coordinates from vehicle JSON file"""
        try:
            with open(gps_path, 'r') as f:
                data = json.load(f)

            lat = data.get('gpsLatitude')
            lon = data.get('gpsLongitude')

            if lat is not None and lon is not None:
                return [float(lat), float(lon)]
            else:
                logger.warning(f"GPS coordinates missing in {gps_path}")
                return None

        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as e:
            logger.error(f"Error extracting GPS from {gps_path}: {e}")
            return None

    def _is_valid_item(self, item: Dict) -> bool:
        """Check if all required files exist and GPS coordinates are available"""
        image_path = Path(item['image_path'])
        gps_path = Path(item['gps_path'])

        return (
            image_path.exists() and
            gps_path.exists() and
            item['ground_truth_coords'] is not None
        )

    def select_images_per_city(self, cities: List[str], images_per_city: int,
                              selection_method: str = "systematic") -> List[Dict]:
        """
        Select specified number of images per city using deterministic method.
        
        Args:
            cities: List of city names to process
            images_per_city: Number of images to select per city
            selection_method: Selection algorithm ("systematic", "first_n", "evenly_spaced")
        
        Returns:
            List of selected item dictionaries
        """
        if not self.discovered_items:
            self.discover_all_triplets()

        selected_items = []

        for city in cities:
            city_items = self.items_by_city.get(city, [])

            if not city_items:
                logger.warning(f"No valid items found for city: {city}")
                continue

            if len(city_items) < images_per_city:
                logger.warning(f"City {city} has only {len(city_items)} valid images, "
                             f"requested {images_per_city}")
                selected = city_items
            else:
                selected = self._select_by_method(city_items, images_per_city, selection_method)

            selected_items.extend(selected)
            logger.info(f"Selected {len(selected)} images from {city}")

        logger.info(f"Total selected: {len(selected_items)} images across {len(cities)} cities")
        return selected_items

    def _select_by_method(self, items: List[Dict], count: int, method: str) -> List[Dict]:
        """Apply selection method to choose images"""
        if method == "first_n":
            return items[:count]

        elif method == "systematic":
            # Select every nth item to get evenly distributed sample
            if len(items) <= count:
                return items

            step = len(items) // count
            selected = []

            for i in range(count):
                idx = i * step
                if idx < len(items):
                    selected.append(items[idx])

            return selected[:count]

        elif method == "evenly_spaced":
            # Select items at evenly spaced intervals
            if len(items) <= count:
                return items

            indices = [int(i * (len(items) - 1) / (count - 1)) for i in range(count)]
            return [items[i] for i in indices]

        else:
            raise ValueError(f"Unknown selection method: {method}")


def clean_item_for_json(item: Dict, dataset: str) -> Dict:
    """Keep only the fields the v2 manifest schema defines."""
    return {
        'image_id': item['image_id'],
        'city': item['city'],
        'dataset': dataset,
        'relpath': item['relpath'],
        'ground_truth_coords': item['ground_truth_coords'],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON dataset files from Cityscapes directory structure"
    )

    parser.add_argument(
        '--cityscapes-root',
        required=True,
        help="Path to Cityscapes dataset root directory"
    )
    parser.add_argument(
        '--dataset',
        default='cityscapes/leftImg8bit/train',
        help="Directory under $VEILLENS_DATA that manifest relpaths resolve against "
             "(default: cityscapes/leftImg8bit/train)"
    )

    parser.add_argument(
        '--output',
        required=True,
        help="Output JSON file path"
    )

    parser.add_argument(
        '--cities',
        help="Comma-separated list of cities to include (default: all available)"
    )

    parser.add_argument(
        '--images-per-city',
        type=int,
        help="Number of images to select per city (default: all available)"
    )

    parser.add_argument(
        '--selection-method',
        choices=['systematic', 'first_n', 'evenly_spaced'],
        default='systematic',
        help="Selection method for choosing images (default: systematic)"
    )

    parser.add_argument(
        '--all-cities',
        action='store_true',
        help="Include all available cities (overrides --cities)"
    )

    parser.add_argument(
        '--list-cities',
        action='store_true',
        help="List available cities and exit"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    try:
        # Initialize discovery
        discovery = CityscapesDiscovery(args.cityscapes_root)

        # Discover all triplets first
        discovery.discover_all_triplets()

        # List cities if requested
        if args.list_cities:
            print("Available cities:")
            for city in sorted(discovery.items_by_city.keys()):
                count = len(discovery.items_by_city[city])
                print(f"  {city}: {count} valid images")
            return

        # Determine which cities to include
        if args.all_cities:
            target_cities = list(discovery.items_by_city.keys())
        elif args.cities:
            target_cities = [city.strip() for city in args.cities.split(',')]
        else:
            target_cities = list(discovery.items_by_city.keys())

        # Validate cities exist
        available_cities = set(discovery.items_by_city.keys())
        invalid_cities = set(target_cities) - available_cities
        if invalid_cities:
            raise ValueError(f"Cities not found: {sorted(invalid_cities)}. "
                           f"Available: {sorted(available_cities)}")

        # Select images
        if args.images_per_city:
            selected_items = discovery.select_images_per_city(
                cities=target_cities,
                images_per_city=args.images_per_city,
                selection_method=args.selection_method
            )
        else:
            # Use all available items from selected cities
            selected_items = []
            for city in target_cities:
                selected_items.extend(discovery.items_by_city[city])

        # Clean items for JSON output. `dataset` is the directory under $VEILLENS_DATA that
        # `relpath` is resolved against, so the manifest carries no absolute paths.
        output_items = [clean_item_for_json(item, args.dataset) for item in selected_items]

        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON file (schema v2: dataset + relpath, resolved against $VEILLENS_DATA)
        with open(output_path, 'w') as f:
            json.dump({"schema_version": 2, "items": output_items}, f, indent=2)

        # Print summary
        print(f"Created dataset JSON: {output_path}")
        print(f"Total images: {len(output_items)}")

        city_counts = defaultdict(int)
        for item in output_items:
            city_counts[item['city']] += 1

        print("Images per city:")
        for city, count in sorted(city_counts.items()):
            print(f"  {city}: {count}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
