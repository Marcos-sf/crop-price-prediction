import React, { useState } from 'react';
import { Container, Box, Typography, Paper, Grid, Card, CardContent, FormControl, InputLabel, Select, MenuItem, Button, List, ListItem, ListItemText, ListItemIcon, Divider } from '@mui/material';
import { Download, PictureAsPdf, TableChart, Assessment } from '@mui/icons-material';

const crops = ['Arecanut', 'Elaichi (Cardamom)', 'Pepper', 'Coconut'];
const mandis = ['Sirsi', 'Shimoga', 'Chikmagalur', 'Madikeri', 'Tiptur', 'Hassan', 'Sullia', 'Tumkur'];

const Export = () => {
  const [selectedCrop, setSelectedCrop] = useState('Arecanut');
  const [selectedMandi, setSelectedMandi] = useState('Sirsi');
  const [loading, setLoading] = useState(false);

  const handleDownloadCSV = async () => {
    setLoading(true);
    try {
      // 1. Fetch historical data
      const histResp = await fetch(`http://localhost:8000/api/v1/history?crop=${selectedCrop.toLowerCase()}&mandi=${selectedMandi.toLowerCase()}`);
      if (!histResp.ok) throw new Error('Could not fetch historical data');
      const histData = await histResp.json();
      // 2. Fetch latest features
      const featuresResp = await fetch(`http://localhost:8000/api/v1/latest-features?crop=${selectedCrop.toLowerCase()}&mandi=${selectedMandi.toLowerCase()}`);
      if (!featuresResp.ok) throw new Error('Could not fetch latest features');
      const features = await featuresResp.json();
      // 3. Call forecast
      const forecastResp = await fetch("http://localhost:8000/api/v1/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop: selectedCrop.toLowerCase(),
          mandi: selectedMandi.toLowerCase(),
          start_date: new Date().toISOString().slice(0, 7) + "-01",
          modal_price_lag1: features.modal_price_lag1,
          modal_price_lag2: features.modal_price_lag2,
          modal_price_lag3: features.modal_price_lag3,
          modal_price_lag5: features.modal_price_lag5,
          modal_price_lag7: features.modal_price_lag7,
          modal_price_lag10: features.modal_price_lag10,
          modal_price_lag14: features.modal_price_lag14,
          modal_price_lag30: features.modal_price_lag30,
          rolling_mean_7: features.rolling_mean_7,
          rolling_mean_30: features.rolling_mean_30,
          rolling_std_7: features.rolling_std_7,
          rolling_std_30: features.rolling_std_30,
          day_of_year: features.day_of_year,
          month: features.month,
          month_sin: features.month_sin,
          month_cos: features.month_cos,
          months: 12
        })
      });
      if (!forecastResp.ok) throw new Error('Could not fetch forecast');
      const forecastData = await forecastResp.json();
      // 4. Combine into CSV
      let csvRows = ["Date,Historical Price,Predicted Price"];
      histData.history.forEach(row => {
        csvRows.push(`${row.date},${row.modal_price},`);
      });
      forecastData.forecast.forEach(row => {
        csvRows.push(`${row.month},,${Math.round(row.predicted_price)}`);
      });
      const csvContent = csvRows.join("\n");
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedCrop}_${selectedMandi}_predictions.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(err.message || 'CSV export failed');
    }
    setLoading(false);
  };

  const handleDownloadExcel = () => {
    // Simulate Excel download
    alert('Excel download functionality will be implemented with backend integration');
  };

  const handleExportChart = () => {
    // Simulate chart export
    alert('Chart export functionality will be implemented with backend integration');
  };

  const handleGenerateReport = () => {
    // Simulate report generation
    alert('Report generation functionality will be implemented with backend integration');
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
        Export & Reports
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" mb={4}>
        Download predictions, export charts, and generate comprehensive reports
      </Typography>

      {/* Selection Controls */}
      <Box display="flex" justifyContent="center" gap={4} mb={4}>
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel>Crop</InputLabel>
          <Select value={selectedCrop} label="Crop" onChange={(e) => setSelectedCrop(e.target.value)}>
            {crops.map(crop => <MenuItem key={crop} value={crop}>{crop}</MenuItem>)}
          </Select>
        </FormControl>
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel>Mandi</InputLabel>
          <Select value={selectedMandi} label="Mandi" onChange={(e) => setSelectedMandi(e.target.value)}>
            {mandis.map(mandi => <MenuItem key={mandi} value={mandi}>{mandi}</MenuItem>)}
          </Select>
        </FormControl>
      </Box>

      {/* Export Options */}
      <Grid container spacing={3}>
        {/* Data Export */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Export
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <Download color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Download as CSV" 
                    secondary="Export prediction data in CSV format"
                  />
                  <Button variant="outlined" size="small" onClick={handleDownloadCSV} disabled={loading}>
                    {loading ? 'Exporting...' : 'Download'}
                  </Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <TableChart color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Download as Excel" 
                    secondary="Export prediction data in Excel format"
                  />
                  <Button variant="outlined" size="small" onClick={handleDownloadExcel}>
                    Download
                  </Button>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Chart Export */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Chart Export
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <PictureAsPdf color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Export Chart as Image" 
                    secondary="Download chart as PNG or JPG"
                  />
                  <Button variant="outlined" size="small" onClick={handleExportChart}>
                    Export
                  </Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <Assessment color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Generate Report" 
                    secondary="Create comprehensive PDF report"
                  />
                  <Button variant="outlined" size="small" onClick={handleGenerateReport}>
                    Generate
                  </Button>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Export History */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Exports
              </Typography>
              <List>
                <ListItem>
                  <ListItemText 
                    primary="Arecanut_Sirsi_predictions.csv" 
                    secondary="Downloaded on July 18, 2025"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText 
                    primary="Pepper_Chikmagalur_chart.png" 
                    secondary="Exported on July 17, 2025"
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText 
                    primary="Coconut_Tiptur_report.pdf" 
                    secondary="Generated on July 16, 2025"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Export Settings */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Export Settings
              </Typography>
              <Box>
                <Typography variant="body2" color="text.secondary" paragraph>
                  • CSV files include historical and predicted prices
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  • Excel files include additional formatting and charts
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  • Image exports are available in PNG and JPG formats
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  • PDF reports include analysis and insights
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Button 
                  variant="contained" 
                  fullWidth 
                  startIcon={<Download />}
                  onClick={handleDownloadCSV}
                  disabled={loading}
                >
                  {loading ? 'Exporting...' : 'Quick CSV Download'}
                </Button>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  startIcon={<PictureAsPdf />}
                  onClick={handleExportChart}
                >
                  Export Current Chart
                </Button>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  startIcon={<Assessment />}
                  onClick={handleGenerateReport}
                >
                  Generate Full Report
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Export; 