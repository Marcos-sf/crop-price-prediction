import React, { useState, useEffect } from 'react';
import { 
  Container, Box, Typography, Paper, Grid, Card, CardContent, FormControl, InputLabel, Select, MenuItem, Button, 
  List, ListItem, ListItemText, ListItemIcon, Divider, Chip, Alert, Snackbar, CircularProgress, Accordion, 
  AccordionSummary, AccordionDetails, TextField, Switch, FormControlLabel, ToggleButtonGroup, ToggleButton
} from '@mui/material';
import { 
  Download, PictureAsPdf, TableChart, Assessment, ExpandMore, FileDownload, 
  TrendingUp, Analytics, BarChart, PieChart, Timeline, CheckCircle, Warning,
  GetApp, Description, Image, DataUsage, Settings
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart as RechartsBarChart, Bar } from 'recharts';

const crops = ['Arecanut', 'Coconut'];
const mandis = ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta', 'Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur'];

const Export = () => {
  const [selectedCrop, setSelectedCrop] = useState('Arecanut');
  const [selectedMandi, setSelectedMandi] = useState('Sirsi');
  const [loading, setLoading] = useState(false);
  const [exportHistory, setExportHistory] = useState([]);
  const [alert, setAlert] = useState({ open: false, message: '', severity: 'info' });
  const [exportData, setExportData] = useState(null);
  const [exportSettings, setExportSettings] = useState({
    includeHistorical: true,
    includeForecast: true,
    includeAnalysis: true,
    includeCharts: true,
    dateRange: '12_months',
    format: 'csv'
  });

  const showAlert = (message, severity = 'info') => {
    setAlert({ open: true, message, severity });
  };

  const handleCloseAlert = () => {
    setAlert({ ...alert, open: false });
  };

  // Load export history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('exportHistory');
    if (savedHistory) {
      setExportHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Save export to history
  const saveToHistory = (exportInfo) => {
    const newHistory = [
      {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        ...exportInfo
      },
      ...exportHistory.slice(0, 9) // Keep only last 10 exports
    ];
    setExportHistory(newHistory);
    localStorage.setItem('exportHistory', JSON.stringify(newHistory));
  };

  // Fetch data for export
  const fetchExportData = async () => {
    setLoading(true);
    try {
      console.log('Fetching export data for:', selectedCrop, selectedMandi);
      
      // Fetch historical data
      const histResp = await fetch(`http://localhost:8000/api/v1/history?crop=${selectedCrop.toLowerCase()}&mandi=${selectedMandi.toLowerCase()}`);
      if (!histResp.ok) {
        const errorText = await histResp.text();
        throw new Error(`Could not fetch historical data: ${histResp.status} - ${errorText}`);
      }
      const histData = await histResp.json();
      
      // Validate historical data
      if (!histData || !histData.history || !Array.isArray(histData.history)) {
        throw new Error('Invalid historical data format received');
      }
      console.log('Historical data received:', histData.history.length, 'records');

      // Fetch latest features
      const featuresResp = await fetch(`http://localhost:8000/api/v1/latest-features?crop=${selectedCrop.toLowerCase()}&mandi=${selectedMandi.toLowerCase()}`);
      if (!featuresResp.ok) {
        const errorText = await featuresResp.text();
        throw new Error(`Could not fetch latest features: ${featuresResp.status} - ${errorText}`);
      }
      const features = await featuresResp.json();
      console.log('Features received:', features);

      // Fetch forecast data
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
          rolling_mean_7: features.rolling_mean_7,
          rolling_std_7: features.rolling_std_7,
          day_of_year: features.day_of_year,
          month: features.month,
          month_sin: features.month_sin,
          month_cos: features.month_cos,
          months: 12
        })
      });
      if (!forecastResp.ok) {
        const errorText = await forecastResp.text();
        throw new Error(`Could not fetch forecast data: ${forecastResp.status} - ${errorText}`);
      }
      const forecastData = await forecastResp.json();
      
      // Validate forecast data
      if (!forecastData || !forecastData.forecast || !Array.isArray(forecastData.forecast)) {
        throw new Error('Invalid forecast data format received');
      }
      console.log('Forecast data received:', forecastData.forecast.length, 'records');

      // Debug: Log the data structure
      console.log('Forecast data structure:', forecastData.forecast[0]);
      console.log('Historical data structure:', histData.history[0]);

      // Create export data object
      const exportDataObj = {
        historical: histData.history || [],
        forecast: forecastData.forecast || [],
        features: features || {},
        metadata: {
          crop: selectedCrop,
          mandi: selectedMandi,
          generatedAt: new Date().toISOString(),
          totalRecords: (histData.history?.length || 0) + (forecastData.forecast?.length || 0)
        }
      };

      console.log('Setting export data:', exportDataObj);
      setExportData(exportDataObj);

      return { histData, forecastData, features };
    } catch (error) {
      console.error('Error fetching export data:', error);
      setExportData(null); // Reset export data on error
      throw error;
    } finally {
      setLoading(false);
    }
  };

  // Generate CSV content
  const generateCSV = (data) => {
    const { historical, forecast, metadata } = data;
    
    let csvRows = [
      `Crop Price Prediction Report - ${metadata.crop} in ${metadata.mandi}`,
      `Generated on: ${new Date(metadata.generatedAt).toLocaleString()}`,
      `Total Records: ${metadata.totalRecords}`,
      '',
      'Date,Historical Price,Predicted Price,Data Type'
    ];

    // Add historical data
    if (exportSettings.includeHistorical) {
      historical.forEach(row => {
        // Ensure date is properly formatted
        const dateStr = row.date ? row.date : 'Unknown Date';
        csvRows.push(`"${dateStr}",${row.modal_price},,Historical`);
      });
    }

    // Add forecast data
    if (exportSettings.includeForecast) {
      forecast.forEach(row => {
        // Use month field for forecast data (as returned by API)
        const dateStr = row.date || row.month || 'Unknown Date';
        const predictedPrice = row.predicted_price ? Math.round(row.predicted_price) : 'N/A';
        csvRows.push(`"${dateStr}",,${predictedPrice},Predicted`);
      });
    }

    return csvRows.join('\n');
  };

  // Generate Excel-like content (CSV with multiple sheets)
  const generateExcel = (data) => {
    const { historical, forecast, metadata } = data;
    
    // Main data sheet
    let mainSheet = [
      `Crop Price Prediction Report - ${metadata.crop} in ${metadata.mandi}`,
      `Generated on: ${new Date(metadata.generatedAt).toLocaleString()}`,
      '',
      'Date,Historical Price,Predicted Price,Data Type'
    ];

    if (exportSettings.includeHistorical) {
      historical.forEach(row => {
        const dateStr = row.date ? row.date : 'Unknown Date';
        mainSheet.push(`"${dateStr}",${row.modal_price},,Historical`);
      });
    }

    if (exportSettings.includeForecast) {
      forecast.forEach(row => {
        const dateStr = row.date || row.month || 'Unknown Date';
        const predictedPrice = row.predicted_price ? Math.round(row.predicted_price) : 'N/A';
        mainSheet.push(`"${dateStr}",,${predictedPrice},Predicted`);
      });
    }

    // Summary sheet
    const prices = historical.map(h => h.modal_price);
    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);

    let summarySheet = [
      'Summary Statistics',
      '',
      'Metric,Value',
      `Average Historical Price,${avgPrice.toFixed(2)}`,
      `Maximum Historical Price,${maxPrice.toFixed(2)}`,
      `Minimum Historical Price,${minPrice.toFixed(2)}`,
      `Total Historical Records,${historical.length}`,
      `Total Forecast Records,${forecast.length}`,
      '',
      'Forecast Summary',
      'Month,Predicted Price',
      ...forecast.map(f => `${f.month},${Math.round(f.predicted_price)}`)
    ];

    return {
      main: mainSheet.join('\n'),
      summary: summarySheet.join('\n')
    };
  };

  // Download CSV
  const handleDownloadCSV = async () => {
    try {
      if (!exportData) {
        await fetchExportData();
      }
      
      const csvContent = generateCSV(exportData);
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedCrop}_${selectedMandi}_predictions.csv`;
      link.click();
      window.URL.revokeObjectURL(url);

      saveToHistory({
        type: 'CSV Export',
        filename: `${selectedCrop}_${selectedMandi}_predictions.csv`,
        crop: selectedCrop,
        mandi: selectedMandi
      });

      showAlert('CSV file downloaded successfully!', 'success');
    } catch (error) {
      showAlert(`CSV export failed: ${error.message}`, 'error');
    }
  };

  // Download Excel (multi-sheet CSV)
  const handleDownloadExcel = async () => {
    try {
      if (!exportData) {
        await fetchExportData();
      }
      
      const excelData = generateExcel(exportData);
      
      // Create zip file with multiple CSV sheets
      const mainBlob = new Blob([excelData.main], { type: 'text/csv;charset=utf-8;' });
      const summaryBlob = new Blob([excelData.summary], { type: 'text/csv;charset=utf-8;' });
      
      // Download main sheet
      const url = window.URL.createObjectURL(mainBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedCrop}_${selectedMandi}_data.csv`;
      link.click();
      window.URL.revokeObjectURL(url);

      // Download summary sheet
      const summaryUrl = window.URL.createObjectURL(summaryBlob);
      const summaryLink = document.createElement('a');
      summaryLink.href = summaryUrl;
      summaryLink.download = `${selectedCrop}_${selectedMandi}_summary.csv`;
      summaryLink.click();
      window.URL.revokeObjectURL(summaryUrl);

      saveToHistory({
        type: 'Excel Export',
        filename: `${selectedCrop}_${selectedMandi}_data.csv + summary.csv`,
        crop: selectedCrop,
        mandi: selectedMandi
      });

      showAlert('Excel files downloaded successfully!', 'success');
    } catch (error) {
      showAlert(`Excel export failed: ${error.message}`, 'error');
    }
  };

  // Export chart as image
  const handleExportChart = async () => {
    try {
      console.log('Starting chart export...');
      console.log('Current exportData:', exportData);
      
      if (!exportData) {
        console.log('No export data, fetching...');
        await fetchExportData();
        // Wait a bit for state to update
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Ensure we have real data
      if (!exportData || !exportData.historical || !exportData.forecast) {
        throw new Error('No valid export data available. Please ensure you have selected a crop and mandi, and the data is loaded.');
      }

      // Use real data for chart
      const historicalData = exportData.historical.slice(-12).map(h => ({
        date: h.date || 'Unknown',
        price: parseFloat(h.modal_price) || 0,
        type: 'Historical'
      })).filter(d => d.price > 0);

      const forecastData = exportData.forecast.map(f => ({
        date: f.date || f.month || 'Unknown',
        price: parseFloat(f.predicted_price) || 0,
        type: 'Predicted'
      })).filter(d => d.price > 0);
      
      const chartData = [...historicalData, ...forecastData];
      
      if (chartData.length === 0) {
        throw new Error('No valid data available for chart export. Please check if the selected crop and mandi have data.');
      }

      console.log('Chart data prepared:', chartData.length, 'data points');

      // Create canvas and draw chart
      const canvas = document.createElement('canvas');
      canvas.width = 1000;
      canvas.height = 600;
      const ctx = canvas.getContext('2d');

      // Draw background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw border
      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, 0, canvas.width, canvas.height);

      // Draw title
      ctx.fillStyle = '#333333';
      ctx.font = 'bold 24px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${selectedCrop} Price Analysis - ${selectedMandi}`, canvas.width / 2, 40);

      // Draw subtitle
      ctx.font = '14px Arial';
      ctx.fillStyle = '#666666';
      ctx.fillText(`Generated on ${new Date().toLocaleDateString()}`, canvas.width / 2, 65);

      // Calculate chart area
      const margin = { top: 100, right: 100, bottom: 100, left: 100 };
      const chartWidth = canvas.width - margin.left - margin.right;
      const chartHeight = canvas.height - margin.top - margin.bottom;

      // Calculate price range
      const prices = chartData.map(d => d.price);
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const priceRange = maxPrice - minPrice;

      // Draw grid lines
      ctx.strokeStyle = '#f0f0f0';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = margin.top + (i / 5) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(canvas.width - margin.right, y);
        ctx.stroke();
      }

      // Draw price labels
      ctx.fillStyle = '#666666';
      ctx.font = '12px Arial';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 5; i++) {
        const price = maxPrice - (i / 5) * priceRange;
        const y = margin.top + (i / 5) * chartHeight;
        ctx.fillText(`₹${Math.round(price)}`, margin.left - 10, y + 4);
      }

      // Draw data points and lines
      chartData.forEach((point, index) => {
        const x = margin.left + (index / (chartData.length - 1)) * chartWidth;
        const y = margin.top + ((maxPrice - point.price) / priceRange) * chartHeight;
        
        // Draw line to previous point
        if (index > 0) {
          const prevPoint = chartData[index - 1];
          const prevX = margin.left + ((index - 1) / (chartData.length - 1)) * chartWidth;
          const prevY = margin.top + ((maxPrice - prevPoint.price) / priceRange) * chartHeight;
          
          ctx.strokeStyle = point.type === 'Historical' ? '#1976d2' : '#ff9800';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(prevX, prevY);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        
        // Draw data point
        ctx.fillStyle = point.type === 'Historical' ? '#1976d2' : '#ff9800';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw point border
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Draw legend
      const legendY = canvas.height - 60;
      ctx.font = '14px Arial';
      
      // Historical legend
      ctx.fillStyle = '#1976d2';
      ctx.fillRect(margin.left, legendY, 20, 15);
      ctx.fillStyle = '#333333';
      ctx.fillText('Historical Data', margin.left + 30, legendY + 12);
      
      // Predicted legend
      ctx.fillStyle = '#ff9800';
      ctx.fillRect(margin.left + 150, legendY, 20, 15);
      ctx.fillStyle = '#333333';
      ctx.fillText('Predicted Data', margin.left + 180, legendY + 12);

      // Convert to blob and download
      canvas.toBlob((blob) => {
        if (!blob) {
          throw new Error('Failed to generate chart image');
        }
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${selectedCrop}_${selectedMandi}_chart.png`;
        link.click();
        window.URL.revokeObjectURL(url);

        saveToHistory({
          type: 'Chart Export',
          filename: `${selectedCrop}_${selectedMandi}_chart.png`,
          crop: selectedCrop,
          mandi: selectedMandi
        });

        showAlert('Chart exported successfully!', 'success');
      }, 'image/png', 0.9);
    } catch (error) {
      console.error('Chart export error:', error);
      showAlert(`Chart export failed: ${error.message}`, 'error');
    }
  };

  // Generate comprehensive report
  const handleGenerateReport = async () => {
    try {
      if (!exportData) {
        await fetchExportData();
      }

      const { historical, forecast, metadata } = exportData;
      
      // Calculate statistics
      const prices = historical.map(h => h.modal_price);
      const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
      const maxPrice = Math.max(...prices);
      const minPrice = Math.min(...prices);
      const volatility = ((maxPrice - minPrice) / avgPrice) * 100;

      // Generate report content
      const reportContent = `
Crop Price Prediction Report
============================

Crop: ${metadata.crop}
Market: ${metadata.mandi}
Generated: ${new Date(metadata.generatedAt).toLocaleString()}

Historical Data Summary:
- Total Records: ${historical.length}
- Average Price: ₹${avgPrice.toFixed(2)}
- Highest Price: ₹${maxPrice.toFixed(2)}
- Lowest Price: ₹${minPrice.toFixed(2)}
- Price Volatility: ${volatility.toFixed(2)}%

Forecast Summary:
- Forecast Period: 12 months
- Average Predicted Price: ₹${(forecast.reduce((a, b) => a + b.predicted_price, 0) / forecast.length).toFixed(2)}
- Highest Predicted: ₹${Math.max(...forecast.map(f => f.predicted_price)).toFixed(2)}
- Lowest Predicted: ₹${Math.min(...forecast.map(f => f.predicted_price)).toFixed(2)}

Market Insights:
- Trend Direction: ${forecast[forecast.length - 1].predicted_price > forecast[0].predicted_price ? 'Upward' : 'Downward'}
- Market Stability: ${volatility < 15 ? 'Stable' : volatility < 25 ? 'Moderate' : 'Volatile'}
- Recommendation: ${volatility < 15 ? 'Good for investment' : 'Monitor closely'}

Detailed Data:
${historical.slice(-10).map(h => `${h.date || 'Unknown Date'}: ₹${h.modal_price}`).join('\n')}

Forecast Data:
${forecast.map(f => `${f.date || f.month || 'Unknown Date'}: ₹${f.predicted_price.toFixed(2)}`).join('\n')}
      `;

      const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedCrop}_${selectedMandi}_report.txt`;
      link.click();
      window.URL.revokeObjectURL(url);

      saveToHistory({
        type: 'Report Generation',
        filename: `${selectedCrop}_${selectedMandi}_report.txt`,
        crop: selectedCrop,
        mandi: selectedMandi
      });

      showAlert('Report generated successfully!', 'success');
    } catch (error) {
      showAlert(`Report generation failed: ${error.message}`, 'error');
    }
  };

  // Quick export all
  const handleQuickExportAll = async () => {
    try {
      await fetchExportData();
      await handleDownloadCSV();
      await handleExportChart();
      await handleGenerateReport();
      showAlert('All exports completed successfully!', 'success');
    } catch (error) {
      showAlert(`Quick export failed: ${error.message}`, 'error');
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
        Export & Reports Center
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" mb={4}>
        Download comprehensive data, charts, and reports for {selectedCrop} in {selectedMandi}
      </Typography>

      {/* Selection Controls */}
      <Box display="flex" justifyContent="center" gap={4} mb={4} flexWrap="wrap">
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
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleQuickExportAll}
          disabled={loading}
          startIcon={<GetApp />}
        >
          {loading ? <CircularProgress size={20} /> : 'Export All'}
        </Button>
      </Box>

      {/* Export Settings */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Settings color="primary" />
                Export Settings
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={exportSettings.includeHistorical}
                        onChange={(e) => setExportSettings({...exportSettings, includeHistorical: e.target.checked})}
                      />
                    }
                    label="Include Historical Data"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={exportSettings.includeForecast}
                        onChange={(e) => setExportSettings({...exportSettings, includeForecast: e.target.checked})}
                      />
                    }
                    label="Include Forecast Data"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={exportSettings.includeAnalysis}
                        onChange={(e) => setExportSettings({...exportSettings, includeAnalysis: e.target.checked})}
                      />
                    }
                    label="Include Analysis"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={exportSettings.includeCharts}
                        onChange={(e) => setExportSettings({...exportSettings, includeCharts: e.target.checked})}
                      />
                    }
                    label="Include Charts"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Export Options */}
      <Grid container spacing={3}>
        {/* Data Export */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DataUsage color="primary" />
                Data Export
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <Download color="success" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Download as CSV" 
                    secondary="Export prediction data in CSV format with historical and forecast data"
                  />
                  <Button variant="contained" size="small" onClick={handleDownloadCSV} disabled={loading}>
                    {loading ? <CircularProgress size={16} /> : 'Download'}
                  </Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <TableChart color="info" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Download as Excel" 
                    secondary="Export with multiple sheets including summary statistics"
                  />
                  <Button variant="contained" size="small" onClick={handleDownloadExcel} disabled={loading}>
                    {loading ? <CircularProgress size={16} /> : 'Download'}
                  </Button>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Visual Export */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Image color="primary" />
                Visual Export
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <BarChart color="success" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Export Chart as Image" 
                    secondary="Download price trend chart as high-quality PNG"
                  />
                  <Button variant="contained" size="small" onClick={handleExportChart} disabled={loading}>
                    {loading ? <CircularProgress size={16} /> : 'Export'}
                  </Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <Description color="warning" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Generate Report" 
                    secondary="Create comprehensive text report with analysis"
                  />
                  <Button variant="contained" size="small" onClick={handleGenerateReport} disabled={loading}>
                    {loading ? <CircularProgress size={16} /> : 'Generate'}
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
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Timeline color="primary" />
                Recent Exports
              </Typography>
              {exportHistory.length > 0 ? (
              <List>
                  {exportHistory.map((exportItem, index) => (
                    <React.Fragment key={exportItem.id}>
                <ListItem>
                        <ListItemIcon>
                          <CheckCircle color="success" />
                        </ListItemIcon>
                  <ListItemText 
                          primary={exportItem.filename}
                          secondary={`${exportItem.type} - ${exportItem.crop} in ${exportItem.mandi} - ${new Date(exportItem.timestamp).toLocaleString()}`}
                        />
                        <Chip 
                          label={exportItem.type} 
                          size="small" 
                          color="primary" 
                          variant="outlined"
                  />
                </ListItem>
                      {index < exportHistory.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
              </List>
              ) : (
                <Box textAlign="center" py={3}>
                  <Typography variant="body2" color="text.secondary">
                    No exports yet. Start by downloading your first report!
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Export Features */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Analytics color="primary" />
                Export Features
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Download color="success" />
                        CSV Export Features
                </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="Historical price data with dates" />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="12-month price forecasts" />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="Data type indicators (Historical/Predicted)" />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon>
                            <CheckCircle color="success" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText primary="Metadata and generation timestamp" />
                        </ListItem>
                      </List>
                    </AccordionDetails>
                  </Accordion>
        </Grid>
        <Grid item xs={12} md={6}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <BarChart color="info" />
                        Chart Export Features
              </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="High-resolution PNG images" />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="Historical vs predicted price visualization" />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="Color-coded data points" />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><CheckCircle color="success" fontSize="small" /></ListItemIcon>
                          <ListItemText primary="Professional chart styling" />
                        </ListItem>
                      </List>
                    </AccordionDetails>
                  </Accordion>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alert System */}
      <Snackbar
        open={alert.open}
        autoHideDuration={6000}
        onClose={handleCloseAlert}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseAlert} severity={alert.severity} sx={{ width: '100%' }}>
          {alert.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Export; 