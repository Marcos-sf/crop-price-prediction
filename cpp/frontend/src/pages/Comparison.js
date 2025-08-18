import React, { useState } from 'react';
import { Container, Box, Typography, Paper, Grid, Card, CardContent, FormControl, InputLabel, Select, MenuItem, Button, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const crops = ['Arecanut', 'Coconut'];
const mandis = ['Sirsi', 'Shimoga', 'Yellapur', 'Siddapur', 'Sagar', 'Kumta', 'Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur'];

const Comparison = () => {
  const [selectedCrop, setSelectedCrop] = useState('Arecanut');
  const [selectedMandis, setSelectedMandis] = useState(['Sirsi', 'Shimoga']);
  const [loading, setLoading] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparisonData, setComparisonData] = useState([]);
  const [summaryStats, setSummaryStats] = useState({});

  const handleMandiChange = (event) => {
    const value = event.target.value;
    if (typeof value === 'string') {
      setSelectedMandis(value.split(','));
    } else {
      setSelectedMandis(value);
    }
  };

  const handleCompare = async () => {
    setLoading(true);
    setComparisonResult(null);
    setComparisonData([]);
    setSummaryStats({});
    try {
      // For each mandi, fetch latest features, then forecast
      const mandiForecasts = await Promise.all(selectedMandis.map(async mandi => {
        // 1. Fetch latest features
        const featuresResp = await fetch(`http://localhost:8000/api/v1/latest-features?crop=${selectedCrop.toLowerCase()}&mandi=${mandi.toLowerCase()}`);
        if (!featuresResp.ok) throw new Error(`Could not fetch features for ${mandi}`);
        const features = await featuresResp.json();
        // 2. Call forecast
        const forecastResp = await fetch("http://localhost:8000/api/v1/forecast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop: selectedCrop.toLowerCase(),
            mandi: mandi.toLowerCase(),
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
        if (!forecastResp.ok) throw new Error(`Could not fetch forecast for ${mandi}`);
        const forecastData = await forecastResp.json();
        return { mandi, forecast: forecastData.forecast };
      }));
      // Align by month
      const allMonths = mandiForecasts[0].forecast.map(item => item.month);
      const compData = allMonths.map((month, idx) => {
        const row = { month };
        mandiForecasts.forEach(({ mandi, forecast }) => {
          row[mandi] = forecast[idx] ? Math.round(forecast[idx].predicted_price) : null;
        });
        return row;
      });
      setComparisonData(compData);
      // Compute summary stats for each mandi
      const stats = {};
      selectedMandis.forEach(mandi => {
        const prices = compData.map(row => row[mandi]).filter(v => v != null);
        if (prices.length > 0) {
          stats[mandi] = {
            avg: Math.round(prices.reduce((a, b) => a + b, 0) / prices.length),
            max: Math.max(...prices),
            min: Math.min(...prices)
          };
        } else {
          stats[mandi] = { avg: '--', max: '--', min: '--' };
        }
      });
      setSummaryStats(stats);
      setComparisonResult({
        crop: selectedCrop,
        mandis: selectedMandis,
        timestamp: new Date().toLocaleString()
      });
    } catch (err) {
      setComparisonResult(null);
      setComparisonData([]);
      setSummaryStats({});
      alert(err.message || 'Comparison failed');
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
        Mandi Comparison
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" mb={4}>
        Compare predicted prices for the next 12 months across different mandis
      </Typography>
      {/* Selection Controls */}
      <Box display="flex" justifyContent="center" gap={4} mb={4}>
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel>Crop</InputLabel>
          <Select value={selectedCrop} label="Crop" onChange={(e) => setSelectedCrop(e.target.value)}>
            {crops.map(crop => <MenuItem key={crop} value={crop}>{crop}</MenuItem>)}
          </Select>
        </FormControl>
        <FormControl sx={{ minWidth: 300 }}>
          <InputLabel>Select Mandis (Max 3)</InputLabel>
          <Select
            multiple
            value={selectedMandis}
            label="Select Mandis (Max 3)"
            onChange={handleMandiChange}
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value) => (
                  <Chip key={value} label={value} size="small" />
                ))}
              </Box>
            )}
          >
            {mandis.map(mandi => <MenuItem key={mandi} value={mandi}>{mandi}</MenuItem>)}
          </Select>
        </FormControl>
        <Button 
          variant="contained" 
          color="primary"
          onClick={handleCompare}
          disabled={loading || selectedMandis.length === 0}
        >
          {loading ? 'Comparing...' : 'Compare'}
        </Button>
      </Box>
      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <Typography variant="h6" color="text.secondary">
            Comparing {selectedCrop} prices across {selectedMandis.length} mandis...
          </Typography>
        </Box>
      ) : comparisonResult ? (
        <>
          <Box mb={3} p={2} bgcolor="info.light" borderRadius={1}>
            <Typography variant="body2" color="white">
              Comparison completed for {comparisonResult.crop} across {comparisonResult.mandis.join(', ')} at {comparisonResult.timestamp}
            </Typography>
          </Box>
          {/* Comparison Chart */}
          <Card elevation={3} sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Future Price Comparison Chart (Next 12 Months)
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {selectedMandis.map((mandi, index) => (
                    <Line
                      key={mandi}
                      type="monotone"
                      dataKey={mandi}
                      stroke={['#1976d2', '#8884d8', '#82ca9d'][index % 3]}
                      strokeWidth={3}
                      dot={{ r: 5 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          {/* Comparison Table */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Future Price Comparison Table (Next 12 Months)
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Month</TableCell>
                          {selectedMandis.map(mandi => (
                            <TableCell key={mandi} align="right">{mandi}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {comparisonData.map((row) => (
                          <TableRow key={row.month}>
                            <TableCell>{row.month}</TableCell>
                            {selectedMandis.map(mandi => (
                              <TableCell key={mandi} align="right">{row[mandi] != null ? `₹${row[mandi]}` : '--'}</TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
            {/* Summary Statistics */}
            <Grid item xs={12} md={4}>
              <Card elevation={3}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Summary Statistics (Predicted)
                  </Typography>
                  {selectedMandis.map((mandi, index) => (
                    <Box key={mandi} sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" fontWeight={600} color="primary">
                        {mandi}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Average Price: ₹{summaryStats[mandi] ? summaryStats[mandi].avg : '--'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Highest: ₹{summaryStats[mandi] ? summaryStats[mandi].max : '--'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Lowest: ₹{summaryStats[mandi] ? summaryStats[mandi].min : '--'}
                      </Typography>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      ) : (
        // Initial state or no data
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <Typography variant="h6" color="text.secondary">
            Please select a crop and mandis to compare predicted prices.
          </Typography>
        </Box>
      )}
    </Container>
  );
};

export default Comparison; 