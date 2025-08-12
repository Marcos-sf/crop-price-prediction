import React, { useState, useEffect } from 'react';
import {
  Container, Box, Typography, FormControl, InputLabel, Select, MenuItem, Button, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, ToggleButton, ToggleButtonGroup, 
  Snackbar, Alert, Grid, Card, CardContent, Chip, Avatar
} from '@mui/material';
import {
  TrendingUp, Agriculture, ShowChart, Update, Psychology,
  BarChart as BarChartIcon, TableChart
} from '@mui/icons-material';
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
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [latestFeatures, setLatestFeatures] = useState(null);
  const [lagLoading, setLagLoading] = useState(false);
  const [predictionRequested, setPredictionRequested] = useState(false);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState('chart');
  const [alert, setAlert] = useState({ open: false, message: '', severity: 'info' });
  const [responseData, setResponseData] = useState(null);

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
    setResponseData(null);
    try {
      const response = await fetch("http://localhost:8000/api/v1/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop: selectedCrop.toLowerCase(),
          mandi: selectedMandi.toLowerCase(),
          start_date: new Date().toISOString().slice(0, 7) + "-01",
          // 11 features as used in training
          modal_price_lag1: latestFeatures.modal_price_lag1,
          modal_price_lag2: latestFeatures.modal_price_lag2,
          modal_price_lag3: latestFeatures.modal_price_lag3,
          modal_price_lag5: latestFeatures.modal_price_lag5,
          modal_price_lag7: latestFeatures.modal_price_lag7,
          rolling_mean_7: latestFeatures.rolling_mean_7,
          rolling_std_7: latestFeatures.rolling_std_7,
          day_of_year: latestFeatures.day_of_year,
          month: latestFeatures.month,
          month_sin: latestFeatures.month_sin,
          month_cos: latestFeatures.month_cos,
          months: 12,
          model_type: selectedModel
        })
      });
      if (!response.ok) {
        throw new Error("Prediction failed");
      }
      const data = await response.json();
      setResponseData(data);
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
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header Section */}
      <Box mb={6} textAlign="center">
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mb: 2 }}>
          <Avatar sx={{ 
            bgcolor: 'primary.main', 
            width: 60, 
            height: 60, 
            mr: 2,
            background: 'linear-gradient(135deg, #2e7d32 0%, #4caf50 100%)'
          }}>
            <TrendingUp sx={{ fontSize: 30 }} />
          </Avatar>
          <Box textAlign="left">
            <Typography variant="h3" fontWeight={700} color="primary.main">
              Price Prediction Dashboard
            </Typography>
            <Typography variant="h6" color="text.secondary">
              AI-Powered Crop Price Forecasting
        </Typography>
          </Box>
        </Box>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Select your crop and market to get intelligent price predictions for the next 12 months using advanced machine learning models
        </Typography>
      </Box>

      {/* Control Panel */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12}>
          <Card elevation={2} sx={{ 
            background: 'linear-gradient(135deg, #ffffff 0%, #f8fffe 100%)',
            border: '1px solid rgba(46, 125, 50, 0.1)'
          }}>
            <CardContent sx={{ p: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Agriculture sx={{ color: 'primary.main', mr: 1 }} />
                <Typography variant="h6" fontWeight={600}>
                  Selection Panel
                </Typography>
                {latestFeatures && (
                  <Chip 
                    label="Data Ready" 
                    color="success" 
                    size="small" 
                    sx={{ ml: 2 }}
                    icon={<Update />}
                  />
                )}
              </Box>
              
                             <Grid container spacing={3} alignItems="center">
                 <Grid item xs={12} sm={3}>
                   <FormControl fullWidth>
                     <InputLabel id="crop-select-label">Select Crop</InputLabel>
          <Select
            labelId="crop-select-label"
            value={selectedCrop}
                       label="Select Crop"
            onChange={e => setSelectedCrop(e.target.value)}
                       sx={{ borderRadius: 2 }}
          >
            {crops.map(crop => (
                         <MenuItem key={crop} value={crop}>
                           <Box sx={{ display: 'flex', alignItems: 'center' }}>
                             <Agriculture sx={{ mr: 1, color: 'primary.main' }} />
                             {crop}
                           </Box>
                         </MenuItem>
            ))}
          </Select>
        </FormControl>
                 </Grid>
                 
                 <Grid item xs={12} sm={3}>
                   <FormControl fullWidth>
                     <InputLabel id="mandi-select-label">Select Market</InputLabel>
          <Select
            labelId="mandi-select-label"
            value={selectedMandi}
                       label="Select Market"
            onChange={e => setSelectedMandi(e.target.value)}
                       sx={{ borderRadius: 2 }}
          >
            {mandis.map(mandi => (
              <MenuItem key={mandi} value={mandi}>{mandi}</MenuItem>
            ))}
          </Select>
        </FormControl>
                 </Grid>
                 
                 <Grid item xs={12} sm={3}>
                   <FormControl fullWidth>
                     <InputLabel id="model-select-label">AI Model</InputLabel>
                     <Select
                       labelId="model-select-label"
                       value={selectedModel}
                       label="AI Model"
                       onChange={e => setSelectedModel(e.target.value)}
                       sx={{ borderRadius: 2 }}
                     >
                       <MenuItem value="ensemble">
                         <Box sx={{ display: 'flex', alignItems: 'center' }}>
                           <Psychology sx={{ mr: 1, color: 'primary.main' }} />
                           Ensemble (LSTM + XGBoost)
                         </Box>
                       </MenuItem>
                       <MenuItem value="lstm">
                         <Box sx={{ display: 'flex', alignItems: 'center' }}>
                           <ShowChart sx={{ mr: 1, color: 'info.main' }} />
                           LSTM Neural Network
                         </Box>
                       </MenuItem>
                       <MenuItem value="xgboost">
                         <Box sx={{ display: 'flex', alignItems: 'center' }}>
                           <BarChartIcon sx={{ mr: 1, color: 'warning.main' }} />
                           XGBoost
                         </Box>
                       </MenuItem>
                     </Select>
                   </FormControl>
                 </Grid>
                 
                 <Grid item xs={12} sm={3}>
        <Button
          variant="contained"
                     size="large"
                     fullWidth
          onClick={handlePredict}
          disabled={loading || lagLoading || !latestFeatures}
                     startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
                     sx={{ 
                       height: 56, 
                       fontWeight: 600,
                       borderRadius: 2,
                       background: 'linear-gradient(135deg, #2e7d32 0%, #4caf50 100%)',
                       '&:hover': {
                         background: 'linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%)',
                       }
                     }}
                   >
                     {loading ? 'Generating...' : lagLoading ? 'Loading Data...' : 'Generate Forecast'}
        </Button>
                 </Grid>
               </Grid>

              {latestFeatures && (
                <Box mt={3} p={2} sx={{ 
                  backgroundColor: 'rgba(46, 125, 50, 0.05)', 
                  borderRadius: 2,
                  border: '1px solid rgba(46, 125, 50, 0.1)'
                }}>
                  <Typography variant="subtitle2" color="primary.main" gutterBottom>
                    Latest Market Data
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">Latest Price</Typography>
                      <Typography variant="h6" fontWeight={600}>
                        ₹{Math.round(latestFeatures.modal_price_lag1)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">7-Day Avg</Typography>
                      <Typography variant="h6" fontWeight={600}>
                        ₹{Math.round(latestFeatures.rolling_mean_7)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">Last Update</Typography>
                      <Typography variant="body2" fontWeight={600}>
                        {latestFeatures.latest_date}
                      </Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="caption" color="text.secondary">Volatility</Typography>
                      <Typography variant="body2" fontWeight={600}>
                        {latestFeatures.rolling_std_7 > 50 ? 'High' : latestFeatures.rolling_std_7 > 20 ? 'Medium' : 'Low'}
                      </Typography>
                    </Grid>
                  </Grid>
      </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      {/* Results Section */}
      {loading ? (
        <Card elevation={2} sx={{ mb: 4 }}>
          <CardContent sx={{ p: 6, textAlign: 'center' }}>
            <Box display="flex" flexDirection="column" alignItems="center">
              <CircularProgress 
                size={60} 
                thickness={4}
                sx={{ 
                  color: 'primary.main',
                  mb: 3
                }}
              />
              <Typography variant="h6" gutterBottom>
                Generating AI Prediction
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Analyzing market data and generating intelligent forecasts...
            </Typography>
          </Box>
          </CardContent>
        </Card>
      ) : predictionRequested && chartData.length > 0 && (
        <>
          {/* Results Header */}
          <Box mb={3}>
            <Typography variant="h5" fontWeight={600} gutterBottom>
              Price Forecast Results
            </Typography>
            <Typography variant="body1" color="text.secondary">
              12-month price prediction for {selectedCrop} in {selectedMandi} market
            </Typography>
          </Box>

          {/* View Toggle */}
          <Box mb={3} display="flex" justifyContent="center">
            <ToggleButtonGroup
              value={view}
              exclusive
              onChange={(_, value) => value && setView(value)}
              aria-label="view selection"
              sx={{ 
                '& .MuiToggleButton-root': {
                  borderRadius: 2,
                  px: 3,
                  py: 1,
                  fontWeight: 600
                }
              }}
            >
              <ToggleButton value="chart" aria-label="Chart View">
                <BarChartIcon sx={{ mr: 1 }} />
                Chart View
              </ToggleButton>
              <ToggleButton value="table" aria-label="Table View">
                <TableChart sx={{ mr: 1 }} />
                Table View
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

                     {/* Chart View */}
          {view === 'chart' && (
             <>
               {/* Main Chart */}
               <Card elevation={2} sx={{ 
                 background: 'linear-gradient(135deg, #ffffff 0%, #f8fffe 100%)',
                 border: '1px solid rgba(46, 125, 50, 0.1)',
                 mb: 3
               }}>
                 <CardContent sx={{ p: 4 }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                       <Box sx={{ display: 'flex', alignItems: 'center' }}>
                         <ShowChart sx={{ color: 'primary.main', mr: 1 }} />
                         <Typography variant="h6" fontWeight={600}>
                           Price Trend Forecast
                         </Typography>
                       </Box>
                       {responseData && responseData.model_type && (
                         <Chip
                           label={`${responseData.model_type.toUpperCase()} Model`}
                           color={
                             responseData.model_type === 'ensemble' ? 'primary' :
                             responseData.model_type === 'lstm' ? 'info' : 'warning'
                           }
                           size="small"
                           sx={{ fontWeight: 600 }}
                         />
                       )}
                     </Box>
                                       <ResponsiveContainer width="100%" height={350}>
                      <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(46, 125, 50, 0.1)" />
                        <XAxis 
                          dataKey="month" 
                          stroke="#666"
                          fontSize={12}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis 
                          stroke="#666"
                          fontSize={12}
                          tick={{ fontSize: 10 }}
                          label={{ value: 'Price (₹)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                        />
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid rgba(46, 125, 50, 0.2)',
                            borderRadius: '8px',
                            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                            fontSize: '12px'
                          }}
                          formatter={(value) => [`₹${value}`, 'Predicted Price']}
                        />
                  <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="predicted" 
                          name="Predicted Price (₹)" 
                          stroke="#2e7d32" 
                          strokeWidth={2} 
                          dot={{ r: 4, fill: '#2e7d32' }}
                          activeDot={{ r: 6, fill: '#4caf50' }}
                        />
                </LineChart>
              </ResponsiveContainer>
                 </CardContent>
               </Card>

               {/* Summary Cards Below Chart */}
               <Grid container spacing={3} mb={4}>
                 <Grid item xs={12} sm={6} md={3}>
                   <Card elevation={1} sx={{ 
                     background: 'linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%)',
                     border: '1px solid rgba(76, 175, 80, 0.2)'
                   }}>
                     <CardContent sx={{ textAlign: 'center', p: 3 }}>
                       <Typography variant="caption" color="text.secondary">
                         Avg Predicted Price
                       </Typography>
                       <Typography variant="h4" fontWeight={700} color="success.main">
                         ₹{Math.round(chartData.reduce((sum, item) => sum + item.predicted, 0) / chartData.length)}
                       </Typography>
                     </CardContent>
                   </Card>
                 </Grid>

                 <Grid item xs={12} sm={6} md={3}>
                   <Card elevation={1} sx={{ 
                     background: 'linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%)',
                     border: '1px solid rgba(255, 152, 0, 0.2)'
                   }}>
                     <CardContent sx={{ textAlign: 'center', p: 3 }}>
                       <Typography variant="caption" color="text.secondary">
                         Highest Price
                       </Typography>
                       <Typography variant="h4" fontWeight={700} color="warning.main">
                         ₹{Math.max(...chartData.map(item => item.predicted))}
                       </Typography>
                     </CardContent>
                   </Card>
                 </Grid>

                 <Grid item xs={12} sm={6} md={3}>
                   <Card elevation={1} sx={{ 
                     background: 'linear-gradient(135deg, #e3f2fd 0%, #e8eaf6 100%)',
                     border: '1px solid rgba(33, 150, 243, 0.2)'
                   }}>
                     <CardContent sx={{ textAlign: 'center', p: 3 }}>
                       <Typography variant="caption" color="text.secondary">
                         Lowest Price
                       </Typography>
                       <Typography variant="h4" fontWeight={700} color="info.main">
                         ₹{Math.min(...chartData.map(item => item.predicted))}
                       </Typography>
                     </CardContent>
                   </Card>
                 </Grid>

                 <Grid item xs={12} sm={6} md={3}>
                   <Card elevation={1} sx={{ 
                     background: 'linear-gradient(135deg, #fce4ec 0%, #f3e5f5 100%)',
                     border: '1px solid rgba(156, 39, 176, 0.2)'
                   }}>
                     <CardContent sx={{ textAlign: 'center', p: 3 }}>
                       <Typography variant="caption" color="text.secondary">
                         Price Volatility
                       </Typography>
                       <Typography variant="h4" fontWeight={700} color="secondary.main">
                         {(() => {
                           const prices = chartData.map(item => item.predicted);
                           const avg = prices.reduce((sum, p) => sum + p, 0) / prices.length;
                           const variance = prices.reduce((sum, p) => sum + Math.pow(p - avg, 2), 0) / prices.length;
                           const volatility = Math.sqrt(variance) / avg * 100;
                           return volatility < 10 ? 'Low' : volatility < 20 ? 'Med' : 'High';
                         })()}
                       </Typography>
                     </CardContent>
                   </Card>
                 </Grid>
               </Grid>
             </>
           )}

          {/* Table View */}
          {view === 'table' && (
            <Card elevation={2} sx={{ mb: 4 }}>
              <CardContent sx={{ p: 0 }}>
                <Box sx={{ p: 3, borderBottom: '1px solid rgba(0,0,0,0.1)' }}>
                  <Typography variant="h6" fontWeight={600}>
                    Detailed Price Forecast
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Monthly predictions for the next 12 months
              </Typography>
                </Box>
                <TableContainer>
                  <Table>
                  <TableHead>
                      <TableRow sx={{ backgroundColor: 'rgba(46, 125, 50, 0.05)' }}>
                        <TableCell sx={{ fontWeight: 600 }}>Month</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>Predicted Price (₹)</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>Change from Previous</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                      {chartData.map((row, index) => {
                        const prevPrice = index > 0 ? chartData[index - 1].predicted : row.predicted;
                        const change = row.predicted - prevPrice;
                        const changePercent = index > 0 ? ((change / prevPrice) * 100).toFixed(1) : 0;
                        
                        return (
                          <TableRow key={row.month} hover>
                            <TableCell sx={{ fontWeight: 500 }}>{row.month}</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
                              ₹{row.predicted}
                            </TableCell>
                            <TableCell align="right">
                              {index > 0 && (
                                <Chip
                                  label={`${change >= 0 ? '+' : ''}${change.toFixed(0)} (${changePercent}%)`}
                                  size="small"
                                  color={change >= 0 ? 'success' : 'error'}
                                  variant="outlined"
                                />
                              )}
                            </TableCell>
                      </TableRow>
                        );
                      })}
                  </TableBody>
                </Table>
              </TableContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* Empty State */}
      {!loading && !predictionRequested && (
        <Card elevation={1} sx={{ 
          background: 'linear-gradient(135deg, #f8fffe 0%, #f0f8ff 100%)',
          border: '2px dashed rgba(46, 125, 50, 0.3)',
          mb: 4
        }}>
          <CardContent sx={{ p: 6, textAlign: 'center' }}>
            <Avatar sx={{ 
              bgcolor: 'rgba(46, 125, 50, 0.1)', 
              width: 80, 
              height: 80, 
              mx: 'auto',
              mb: 3
            }}>
              <TrendingUp sx={{ fontSize: 40, color: 'primary.main' }} />
            </Avatar>
            <Typography variant="h5" fontWeight={600} gutterBottom>
              Ready to Generate Forecast
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 400, mx: 'auto' }}>
              Select your crop and market from the panel above, then click "Generate Forecast" to see intelligent price predictions for the next 12 months.
        </Typography>
          </CardContent>
        </Card>
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