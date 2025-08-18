import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processor for handling historical crop price data."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            current_file = Path(__file__)
            self.data_dir = current_file.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def read_data_file(self, file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        # Try CSV first
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Read {file_path} as CSV")
            return df
        except Exception as e_csv:
            logger.info(f"Could not read {file_path} as CSV: {e_csv}")
        # Try Excel
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            logger.info(f"Read {file_path} as Excel (openpyxl)")
            return df
        except Exception as e_xlsx:
            logger.info(f"Could not read {file_path} as Excel (openpyxl): {e_xlsx}")
        try:
            df = pd.read_excel(file_path, engine='xlrd')
            logger.info(f"Read {file_path} as Excel (xlrd)")
            return df
        except Exception as e_xls:
            logger.error(f"Could not read {file_path} as Excel (xlrd): {e_xls}")
        raise Exception(f"Could not read {file_path} as CSV or Excel.")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        logger.info(f"Original columns: {list(df_clean.columns)}")

        # Explicit mapping for your Agmarknet data
        expected_columns = {
            'date': ['date', 'date_time', 'timestamp', 'price_date', 'arrival_date'],
            'mandi': ['mandi', 'market', 'market_name'],
            'crop': ['crop', 'commodity', 'crop_name'],
            'min_price': ['min_price', 'minimum_price', 'min', 'min_price_(rs./quintal)', 'min_price_rs./quintal', 'min_price_(rs./quintal)'],
            'max_price': ['max_price', 'maximum_price', 'max', 'max_price_(rs./quintal)', 'max_price_rs./quintal', 'max_price_(rs./quintal)'],
            'modal_price': ['modal_price', 'modal', 'price', 'modal_price_(rs./quintal)', 'modal_price_rs./quintal', 'modal_price'],
            'arrival': ['arrival', 'quantity', 'arrival_quantity']
        }

        # Map columns to standard names
        column_mapping = {}
        for standard_name, possible_names in expected_columns.items():
            for col in df_clean.columns:
                for possible in possible_names:
                    if col == possible or col.replace('_', '') == possible.replace('_', ''):
                        column_mapping[col] = standard_name
                        break

        logger.info(f"Column mapping: {column_mapping}")
        df_clean = df_clean.rename(columns=column_mapping)
        logger.info(f"Final columns: {list(df_clean.columns)}")

        # Handle date column
        if 'date' not in df_clean.columns:
            logger.error(f"No date column found! Columns: {list(df_clean.columns)}")
            raise Exception("No date column found in data.")
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])

        # Handle mandi column
        if 'mandi' not in df_clean.columns:
            df_clean['mandi'] = 'unknown'

        # Handle crop column
        if 'crop' not in df_clean.columns:
            df_clean['crop'] = 'unknown'

        # Handle price columns
        for price_col in ['min_price', 'max_price', 'modal_price']:
            if price_col in df_clean.columns:
                df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors='coerce')
            else:
                df_clean[price_col] = np.nan

        # Fill missing modal_price with average of min and max
        mask = df_clean['modal_price'].isna() & df_clean['min_price'].notna() & df_clean['max_price'].notna()
        df_clean.loc[mask, 'modal_price'] = (df_clean.loc[mask, 'min_price'] + df_clean.loc[mask, 'max_price']) / 2

        df_clean['source'] = 'Agmarknet'

        # Keep only relevant columns
        keep_cols = ['date', 'mandi', 'crop', 'min_price', 'max_price', 'modal_price', 'source']
        df_clean = df_clean[[col for col in keep_cols if col in df_clean.columns]]

        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean

    def process_crop_data(self, crop_name: str) -> pd.DataFrame:
        # Look for CSV or XLS/XLSX file in raw directory
        csv_files = list(self.raw_dir.glob(f"*{crop_name}*.csv"))
        xls_files = list(self.raw_dir.glob(f"*{crop_name}*.xls*"))
        data_files = csv_files + xls_files

        if not data_files:
            logger.warning(f"No data file found for {crop_name}")
            return pd.DataFrame()

        file_path = data_files[0]
        df = self.read_data_file(str(file_path))
        df_clean = self.clean_data(df)
        if 'crop' not in df_clean.columns or df_clean['crop'].iloc[0] == 'unknown':
            df_clean['crop'] = crop_name

        output_file = self.processed_dir / f"{crop_name}_processed.csv"
        df_clean.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        return df_clean

    def get_available_crops(self) -> List[str]:
        csv_files = list(self.processed_dir.glob("*_processed.csv"))
        crops = [f.stem.replace('_processed', '') for f in csv_files]
        return crops

    def load_processed_data(self, crop_name: str) -> pd.DataFrame:
        file_path = self.processed_dir / f"{crop_name}_processed.csv"
        if not file_path.exists():
            logger.warning(f"Processed data not found for {crop_name}")
            return pd.DataFrame()
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            logger.error(f"No 'date' column found in processed data for {crop_name}. Columns: {list(df.columns)}")
        return df

    def get_mandis_for_crop(self, crop_name: str) -> List[str]:
        df = self.load_processed_data(crop_name)
        if df.empty or 'mandi' not in df.columns:
            return []
        return df['mandi'].unique().tolist()

    def prepare_training_data(self, crop_name: str, mandi_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        df = self.load_processed_data(crop_name)
        if df.empty or 'modal_price' not in df.columns or 'date' not in df.columns:
            return np.array([]), np.array([])
        if mandi_name and 'mandi' in df.columns:
            df = df[df['mandi'] == mandi_name]
        if df.empty:
            return np.array([]), np.array([])
        df = df.sort_values('date')
        df['price_lag_1'] = df['modal_price'].shift(1)
        df['price_lag_7'] = df['modal_price'].shift(7)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df = df.dropna()
        feature_columns = ['price_lag_1', 'price_lag_7', 'day_of_year', 'month']
        features = df[feature_columns].values
        target = df['modal_price'].values
        logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
        return features, target

def main():
    processor = DataProcessor()
    crops = ['arecanut', 'coconut']
    for crop in crops:
        logger.info(f"Processing {crop} data...")
        try:
            df = processor.process_crop_data(crop)
            if not df.empty:
                logger.info(f"✅ Successfully processed {crop} data")
                logger.info(f"   Shape: {df.shape}")
                logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                mandis = processor.get_mandis_for_crop(crop)
                if mandis:
                    logger.info(f"   Available mandis: {', '.join(mandis)}")
                else:
                    logger.warning(f"   No mandi information found for {crop}")
                features, target = processor.prepare_training_data(crop)
                if len(features) > 0:
                    logger.info(f"   Training data: {features.shape[0]} samples, {features.shape[1]} features")
                else:
                    logger.warning(f"   Could not prepare training data for {crop}")
            else:
                logger.warning(f"❌ No data found for {crop}")
        except Exception as e:
            logger.error(f"❌ Error processing {crop}: {e}")

    logger.info("\n" + "="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    available_crops = processor.get_available_crops()
    if available_crops:
        logger.info(f"✅ Successfully processed crops: {', '.join(available_crops)}")
        for crop in available_crops:
            mandis = processor.get_mandis_for_crop(crop)
            logger.info(f"   {crop}: {len(mandis)} mandis available")
    else:
        logger.warning("❌ No crops were successfully processed")
    logger.info("\nNext steps:")
    logger.info("1. Check the processed data in: backend/app/data/processed/")
    logger.info("2. Run the backend server: python -m uvicorn app.main:app --reload")
    logger.info("3. Test the API endpoints with your real data")

if __name__ == "__main__":
    main()