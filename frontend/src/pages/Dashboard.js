import React, { useState, useEffect } from 'react';
import {
  Container, Box, Typography, FormControl, InputLabel, Select, MenuItem, Button, Paper, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, ToggleButton, ToggleButtonGroup, Snackbar, Alert
} from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const crops = ['Arecanut', 'Coconut'];
const mandis = [
  'Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta',
  'Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur'
];

const Dashboard = () => {
  const [selectedCrop, setSelectedCrop] = useState('Arecanut');
  const [selectedMandi, setSelectedMandi] = useState('Sirsi');
  const [latestFeatures, setLatestFeatures] = useState(null);
  const [lagLoading, setLagLoading] = useState(false);
  const [predictionRequested, setPredictionRequested] = useState(false);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState('chart');
  const [alert, setAlert] = useState({ open: false, message: '', severity: 'info' });

  useEffect(() => {
    const fetchLatestFeatures = async () => {
      setLagLoading(true);
      setLatestFeatures(null);
      try {
        const response = await fetch(
          `http://localhost:8000/api/v1/latest-features?crop=${selectedCrop.toLowerCase()}&mandi=${selectedMandi.toLowerCase()}`
        );
        if (!response.ok) throw new Error("Could not fetch features");
        const data = await response.json();
        setLatestFeatures(data);
      } catch (err) {
        setLatestFeatures(null);
        showAlert("Could not fetch features for this crop/mandi.", "error");
      }
      setLagLoading(false);
    };
    fetchLatestFeatures();
    // eslint-disable-next-line
  }, [selectedCrop, selectedMandi]);

  const showAlert = (message, severity = 'info') => {
    setAlert({ open: true, message, severity });
  };

  const handleCloseAlert = () => {
    setAlert({ ...alert, open: false });
  };

  const handlePredict = async () => {
    if (!latestFeatures) {
      showAlert("Features not loaded yet.", "warning");
      return;
    }
    setLoading(true);
    setPredictionRequested(false);
    setChartData([]);
    try {
      const response = await fetch("http://localhost:8000/api/v1/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop: selectedCrop.toLowerCase(),
          mandi: selectedMandi.toLowerCase(),
          start_date: new Date().toISOString().slice(0, 7) + "-01",
          // 16 features in correct order
          modal_price_lag1: latestFeatures.modal_price_lag1,
          modal_price_lag2: latestFeatures.modal_price_lag2,
          modal_price_lag3: latestFeatures.modal_price_lag3,
          modal_price_lag5: latestFeatures.modal_price_lag5,
          modal_price_lag7: latestFeatures.modal_price_lag7,
          modal_price_lag10: latestFeatures.modal_price_lag10,
          modal_price_lag14: latestFeatures.modal_price_lag14,
          modal_price_lag30: latestFeatures.modal_price_lag30,
          rolling_mean_7: latestFeatures.rolling_mean_7,
          rolling_mean_30: latestFeatures.rolling_mean_30,
          rolling_std_7: latestFeatures.rolling_std_7,
          rolling_std_30: latestFeatures.rolling_std_30,
          day_of_year: latestFeatures.day_of_year,
          month: latestFeatures.month,
          month_sin: latestFeatures.month_sin,
          month_cos: latestFeatures.month_cos,
          months: 12
        })
      });
      if (!response.ok) {
        throw new Error("Prediction failed");
      }
      const data = await response.json();
      const merged = data.forecast.map(item => ({
        month: item.month,
        predicted: Math.round(item.predicted_price)
      }));
      setChartData(merged);
      setPredictionRequested(true);
      showAlert(`Prediction completed for ${selectedCrop} in ${selectedMandi}!`, "success");
    } catch (err) {
      showAlert(err.message || "Prediction failed", "error");
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box mb={4}>
        <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
          Crop Price Prediction Dashboard
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary">
          Select a crop and mandi to view the predicted prices for the next 12 months
        </Typography>
      </Box>
      <Box display="flex" flexWrap="wrap" justifyContent="center" gap={4} mb={4}>
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel id="crop-select-label">Crop</InputLabel>
          <Select
            labelId="crop-select-label"
            value={selectedCrop}
            label="Crop"
            onChange={e => setSelectedCrop(e.target.value)}
          >
            {crops.map(crop => (
              <MenuItem key={crop} value={crop}>{crop}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel id="mandi-select-label">Mandi</InputLabel>
          <Select
            labelId="mandi-select-label"
            value={selectedMandi}
            label="Mandi"
            onChange={e => setSelectedMandi(e.target.value)}
          >
            {mandis.map(mandi => (
              <MenuItem key={mandi} value={mandi}>{mandi}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Button
          variant="contained"
          color="primary"
          onClick={handlePredict}
          sx={{ height: 56, minWidth: 120, fontWeight: 600 }}
          disabled={loading || lagLoading || !latestFeatures}
        >
          {loading ? 'Loading...' : lagLoading ? 'Loading Features...' : 'Predict'}
        </Button>
      </Box>
      <Box mb={2}>
        {lagLoading ? (
          <Typography color="text.secondary">Fetching latest features...</Typography>
        ) : null}
      </Box>
      {loading ? (
        <Paper elevation={4} sx={{ p: { xs: 2, sm: 4 }, minHeight: 340, borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', mb: 4 }}>
          <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" width="100%">
            <CircularProgress color="primary" />
            <Typography variant="body1" color="text.secondary" mt={2}>
              Generating prediction...
            </Typography>
          </Box>
        </Paper>
      ) : predictionRequested && (
        <>
          <Box mb={2} display="flex" justifyContent="center">
            <ToggleButtonGroup
              value={view}
              exclusive
              onChange={(_, value) => value && setView(value)}
              aria-label="view selection"
              size="small"
            >
              <ToggleButton value="chart" aria-label="Chart View">Chart View</ToggleButton>
              <ToggleButton value="table" aria-label="Table View">Table View</ToggleButton>
            </ToggleButtonGroup>
          </Box>
          {view === 'chart' && (
            <Paper elevation={4} sx={{ p: { xs: 2, sm: 4 }, minHeight: 340, borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff', mb: 4 }}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="predicted" name="Predicted Price" stroke="#1976d2" strokeWidth={3} dot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          )}
          {view === 'table' && (
            <Box mb={4}>
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Price Table (Predicted)
              </Typography>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Month</TableCell>
                      <TableCell align="right">Predicted Price</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {chartData.map((row) => (
                      <TableRow key={row.month}>
                        <TableCell>{row.month}</TableCell>
                        <TableCell align="right">{row.predicted}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </>
      )}
      {!loading && !predictionRequested && (
        <Typography variant="h6" color="text.secondary" align="center">
          Select a crop and mandi, then click Predict to see the price forecast for the next 12 months.
        </Typography>
      )}
      <Snackbar
        open={alert.open}
        autoHideDuration={10000}
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

export default Dashboard;