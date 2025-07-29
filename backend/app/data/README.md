# Historical Crop Data

This directory contains historical crop price data for the crop price prediction project.

## Data Structure

### File Organization
```
data/
├── raw/                    # Original XLS files from Agmarknet
│   ├── arecanut.xls
│   ├── coconut.xls
│   └── ...
├── processed/              # Cleaned and processed CSV files
│   ├── arecanut_processed.csv
│   ├── coconut_processed.csv
│   └── ...
└── README.md              # This file
```

### Expected Data Format
The XLS files should contain the following columns:
- Date: Date of the price record
- Mandi: Name of the mandi/market
- Crop: Crop name (arecanut, coconut, etc.)
- Min_Price: Minimum price per quintal
- Max_Price: Maximum price per quintal
- Modal_Price: Modal price per quintal
- Arrival: Quantity arrived (optional)
- Source: Data source (e.g., "Agmarknet")

### Data Processing
- Raw XLS files are converted to CSV format
- Missing values are handled appropriately
- Date formats are standardized
- Price data is cleaned and validated

## Usage
1. Place your XLS files in the `raw/` directory
2. Run the data processing script to convert and clean the data
3. The processed data will be used by the ML models for training and prediction 