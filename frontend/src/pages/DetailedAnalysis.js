import React, { useState } from 'react';
import { 
  Container, Box, Typography, Paper, Grid, Card, CardContent, FormControl, InputLabel, Select, MenuItem, Button, Snackbar, Alert,
  Chip, Divider, List, ListItem, ListItemText, ListItemIcon, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import { 
  TrendingUp, TrendingDown, ShowChart, Analytics, Assessment, Timeline, 
  ExpandMore, Warning, CheckCircle, Info, TrendingFlat, Speed, 
  BarChart as BarChartIcon, PieChart as PieChartIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const crops = ['Arecanut', 'Coconut'];
const mandis = ['Sirsi', 'Shimoga', 'Yellapur', 'Siddapur', 'Sagar', 'Kumta', 'Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur'];

function getQuarter(dateStr) {
  const month = new Date(dateStr).getMonth();
  if (month < 3) return 'Q1';
  if (month < 6) return 'Q2';
  if (month < 9) return 'Q3';
  return 'Q4';
}

const DetailedAnalysis = () => {
  const [selectedCrop, setSelectedCrop] = useState('Arecanut');
  const [selectedMandi, setSelectedMandi] = useState('Sirsi');
  const [loading, setLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [seasonalityData, setSeasonalityData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [advancedMetrics, setAdvancedMetrics] = useState(null);
  const [marketInsights, setMarketInsights] = useState(null);
  const [alert, setAlert] = useState({ open: false, message: '', severity: 'info' });

  const showAlert = (message, severity = 'info') => {
    setAlert({ open: true, message, severity });
  };

  const handleCloseAlert = () => {
    setAlert({ ...alert, open: false });
  };

  const calculateAdvancedMetrics = (history) => {
    if (!history || history.length < 2) return null;

    const prices = history.map(item => item.modal_price);
    const dates = history.map(item => new Date(item.date));
    
    // Calculate price changes
    const priceChanges = [];
    for (let i = 1; i < prices.length; i++) {
      const change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100;
      priceChanges.push(change);
    }

    // Calculate moving averages
    const movingAverages = [];
    const windowSize = Math.min(7, Math.floor(prices.length / 4));
    for (let i = windowSize - 1; i < prices.length; i++) {
      const avg = prices.slice(i - windowSize + 1, i + 1).reduce((a, b) => a + b, 0) / windowSize;
      movingAverages.push({ date: dates[i], avg });
    }

    // Calculate volatility metrics
    const volatility = Math.std(priceChanges);
    const maxGain = Math.max(...priceChanges);
    const maxLoss = Math.min(...priceChanges);
    const avgGain = priceChanges.filter(x => x > 0).reduce((a, b) => a + b, 0) / priceChanges.filter(x => x > 0).length || 0;
    const avgLoss = priceChanges.filter(x => x < 0).reduce((a, b) => a + b, 0) / priceChanges.filter(x => x < 0).length || 0;

    // Calculate trend strength
    const recentPrices = prices.slice(-Math.min(30, prices.length));
    const trendStrength = recentPrices.length > 1 ? 
      (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] * 100 : 0;

    // Calculate seasonal patterns
    const monthlyAverages = {};
    history.forEach(item => {
      const month = new Date(item.date).getMonth();
      if (!monthlyAverages[month]) monthlyAverages[month] = [];
      monthlyAverages[month].push(item.modal_price);
    });

    const seasonalPattern = Object.entries(monthlyAverages).map(([month, prices]) => ({
      month: new Date(2024, parseInt(month), 1).toLocaleString('default', { month: 'long' }),
      avgPrice: prices.reduce((a, b) => a + b, 0) / prices.length
    }));

    return {
      priceChanges,
      movingAverages,
      volatility,
      maxGain,
      maxLoss,
      avgGain,
      avgLoss,
      trendStrength,
      seasonalPattern,
      totalDays: history.length,
      priceRange: Math.max(...prices) - Math.min(...prices),
      priceRangePercentage: ((Math.max(...prices) - Math.min(...prices)) / Math.min(...prices)) * 100
    };
  };

  const generateMarketInsights = (history, metrics, advancedMetrics) => {
    if (!history || !metrics || !advancedMetrics) return null;

    const insights = {
      marketHealth: '',
      riskAssessment: '',
      opportunityAnalysis: '',
      supplyDemandFactors: [],
      externalFactors: [],
      recommendations: []
    };

    // Market Health Assessment
    if (metrics.volatility < 10) {
      insights.marketHealth = 'Excellent - Low volatility indicates stable market conditions';
    } else if (metrics.volatility < 20) {
      insights.marketHealth = 'Good - Moderate volatility with manageable risk';
    } else if (metrics.volatility < 30) {
      insights.marketHealth = 'Fair - High volatility requires careful monitoring';
    } else {
      insights.marketHealth = 'Poor - Very high volatility indicates unstable market';
    }

    // Risk Assessment
    if (advancedMetrics.volatility > 25) {
      insights.riskAssessment = 'High Risk - Significant price fluctuations detected';
    } else if (advancedMetrics.volatility > 15) {
      insights.riskAssessment = 'Medium Risk - Moderate price volatility';
    } else {
      insights.riskAssessment = 'Low Risk - Stable price movements';
    }

    // Opportunity Analysis
    if (metrics.trendDirection === 'Upward' && advancedMetrics.trendStrength > 5) {
      insights.opportunityAnalysis = 'Strong Buy Opportunity - Clear upward trend with momentum';
    } else if (metrics.trendDirection === 'Upward') {
      insights.opportunityAnalysis = 'Moderate Buy Opportunity - Gradual upward movement';
    } else if (metrics.trendDirection === 'Downward' && advancedMetrics.trendStrength < -5) {
      insights.opportunityAnalysis = 'Sell Opportunity - Declining trend detected';
    } else {
      insights.opportunityAnalysis = 'Hold - Stable market conditions';
    }

    // Supply-Demand Factors
    const seasonalHigh = advancedMetrics.seasonalPattern.reduce((max, item) => 
      item.avgPrice > max.avgPrice ? item : max
    );
    const seasonalLow = advancedMetrics.seasonalPattern.reduce((min, item) => 
      item.avgPrice < min.avgPrice ? item : min
    );

    insights.supplyDemandFactors = [
      `Peak season: ${seasonalHigh.month} (₹${seasonalHigh.avgPrice.toFixed(0)})`,
      `Low season: ${seasonalLow.month} (₹${seasonalLow.avgPrice.toFixed(0)})`,
      `Price range: ₹${advancedMetrics.priceRange.toFixed(0)} (${advancedMetrics.priceRangePercentage.toFixed(1)}%)`,
      `Average daily change: ${(advancedMetrics.avgGain + advancedMetrics.avgLoss).toFixed(2)}%`
    ];

    // External Factors
    insights.externalFactors = [
      'Monsoon patterns affecting crop yield',
      'Government policies and MSP changes',
      'Export demand fluctuations',
      'Transportation costs and logistics',
      'Competing crop prices',
      'Storage and warehousing capacity'
    ];

    // Recommendations
    if (metrics.trendDirection === 'Upward' && advancedMetrics.trendStrength > 10) {
      insights.recommendations = [
        'Consider increasing procurement during current favorable prices',
        'Monitor supply chain for optimal timing',
        'Evaluate storage options for future sales',
        'Track competitor market movements'
      ];
    } else if (metrics.trendDirection === 'Downward') {
      insights.recommendations = [
        'Consider holding inventory until market stabilizes',
        'Explore alternative markets or crops',
        'Focus on cost optimization',
        'Monitor for market reversal signals'
      ];
    } else {
      insights.recommendations = [
        'Maintain current market position',
        'Diversify across multiple mandis',
        'Implement risk management strategies',
        'Stay updated with market news and policies'
      ];
    }

    return insights;
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setAnalysisData(null);
    setTrendData([]);
    setSeasonalityData([]);
    setMetrics(null);
    setAdvancedMetrics(null);
    setMarketInsights(null);
    
    try {
      const response = await fetch(`http://localhost:8000/api/v1/history?crop=${selectedCrop.toLowerCase()}&mandi=${selectedMandi.toLowerCase()}`);
      if (!response.ok) throw new Error('Could not fetch historical data');
      const data = await response.json();
      const history = data.history;
      if (!history || history.length === 0) throw new Error('No data available');
      
      // Trend: last 12 months (or all if <12)
      const trend = history.slice(-12).map(item => ({
        month: new Date(item.date).toLocaleString('default', { month: 'short', year: 'numeric' }),
        price: item.modal_price
      }));
      setTrendData(trend);
      
      // Seasonality: average price per quarter
      const byQuarter = {};
      history.forEach(item => {
        const year = new Date(item.date).getFullYear();
        const q = getQuarter(item.date);
        const key = `${q} ${year}`;
        if (!byQuarter[key]) byQuarter[key] = [];
        byQuarter[key].push(item.modal_price);
      });
      const seasonality = Object.entries(byQuarter).map(([season, prices]) => ({
        season,
        avgPrice: prices.reduce((a, b) => a + b, 0) / prices.length
      })).sort((a, b) => a.season.localeCompare(b.season));
      setSeasonalityData(seasonality);
      
      // Key metrics
      const prices = history.map(item => item.modal_price);
      const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
      const maxPrice = Math.max(...prices);
      const minPrice = Math.min(...prices);
      const volatility = ((maxPrice - minPrice) / avgPrice) * 100;
      const trendDirection = trend.length > 1 ? (trend[trend.length - 1].price > trend[0].price ? 'Upward' : 'Downward') : 'Stable';
      
      const basicMetrics = {
        avgPrice: avgPrice.toFixed(2),
        volatility: volatility.toFixed(2),
        trendDirection,
        confidence: '85%',
        maxPrice: maxPrice.toFixed(2),
        minPrice: minPrice.toFixed(2),
        totalRecords: history.length
      };
      setMetrics(basicMetrics);
      
      // Advanced metrics
      const advanced = calculateAdvancedMetrics(history);
      setAdvancedMetrics(advanced);
      
      // Market insights
      const insights = generateMarketInsights(history, basicMetrics, advanced);
      setMarketInsights(insights);
      
      setAnalysisData({
        crop: selectedCrop,
        mandi: selectedMandi,
        timestamp: new Date().toLocaleString()
      });
      
      showAlert(`Analysis completed for ${selectedCrop} in ${selectedMandi}!`, 'success');
    } catch (err) {
      showAlert(err.message || 'Analysis failed', 'error');
    }
    setLoading(false);
  };

  // Add Math.std function for standard deviation
  Math.std = function(array) {
    const n = array.length;
    const mean = array.reduce((a, b) => a + b) / n;
    return Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  return (
    <Container maxWidth={false} sx={{ py: 4, px: 2 }}>
      <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
        Detailed Market Analysis
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" mb={4}>
        Comprehensive analysis of crop price trends, volatility, and market insights
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
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleAnalyze}
          disabled={loading}
          startIcon={<Analytics />}
        >
          {loading ? 'Analyzing...' : 'Analyze Market'}
        </Button>
      </Box>

      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <Typography variant="h6" color="text.secondary">
            Analyzing {selectedCrop} prices in {selectedMandi}...
          </Typography>
        </Box>
      ) : (
        <>
          {analysisData && (
            <Box mb={3} p={2} bgcolor="success.light" borderRadius={1}>
              <Typography variant="body2" color="white">
                Analysis completed for {analysisData.crop} in {analysisData.mandi} at {analysisData.timestamp}
              </Typography>
            </Box>
          )}

          <Grid container spacing={3}>
            {/* Market Health Overview */}
            {marketInsights && (
              <Grid item xs={12}>
                <Card elevation={3}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Assessment color="primary" />
                      Market Health Overview
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Box p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="subtitle2" color="text.secondary">Market Health</Typography>
                          <Chip 
                            label={marketInsights.marketHealth.split(' - ')[0]} 
                            color={marketInsights.marketHealth.includes('Excellent') ? 'success' : 
                                   marketInsights.marketHealth.includes('Good') ? 'primary' :
                                   marketInsights.marketHealth.includes('Fair') ? 'warning' : 'error'}
                            sx={{ mt: 1 }}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="subtitle2" color="text.secondary">Risk Assessment</Typography>
                          <Chip 
                            label={marketInsights.riskAssessment.split(' - ')[0]} 
                            color={marketInsights.riskAssessment.includes('Low') ? 'success' : 
                                   marketInsights.riskAssessment.includes('Medium') ? 'warning' : 'error'}
                            sx={{ mt: 1 }}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="subtitle2" color="text.secondary">Opportunity</Typography>
                          <Chip 
                            label={marketInsights.opportunityAnalysis.split(' - ')[0]} 
                            color={marketInsights.opportunityAnalysis.includes('Buy') ? 'success' : 
                                   marketInsights.opportunityAnalysis.includes('Hold') ? 'primary' : 'warning'}
                            sx={{ mt: 1 }}
                          />
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Price Trend Analysis */}
            <Grid item xs={12} lg={8}>
              <Card elevation={3}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ShowChart color="primary" />
                    Price Trend Analysis
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={trendData} margin={{ top: 20, right: 40, left: 30, bottom: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip formatter={(value) => [`₹${value}`, 'Price']} />
                      <Legend />
                      <Line type="monotone" dataKey="price" stroke="#1976d2" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Advanced Metrics */}
            {advancedMetrics && (
              <Grid item xs={12} lg={4}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Speed color="primary" />
                      Advanced Metrics
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemIcon>
                          <TrendingUp color="success" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Max Gain" 
                          secondary={`${advancedMetrics.maxGain.toFixed(2)}%`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <TrendingDown color="error" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Max Loss" 
                          secondary={`${advancedMetrics.maxLoss.toFixed(2)}%`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <Analytics color="info" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Volatility" 
                          secondary={`${advancedMetrics.volatility.toFixed(2)}%`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <Timeline color="warning" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Trend Strength" 
                          secondary={`${advancedMetrics.trendStrength.toFixed(2)}%`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon>
                          <Info color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary="Price Range" 
                          secondary={`₹${advancedMetrics.priceRange.toFixed(0)}`} 
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Seasonal Analysis */}
            <Grid item xs={12} lg={6}>
              <Card elevation={3}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <BarChartIcon color="primary" />
                    Seasonal Price Patterns
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={seasonalityData} margin={{ top: 20, right: 40, left: 30, bottom: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="season" />
                      <YAxis />
                      <Tooltip formatter={(value) => [`₹${value}`, 'Average Price']} />
                      <Legend />
                      <Bar dataKey="avgPrice" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Monthly Seasonal Pattern */}
            {advancedMetrics && (
              <Grid item xs={12} lg={6}>
                <Card elevation={3}>
                  <CardContent sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <PieChartIcon color="primary" />
                      Monthly Price Distribution
                    </Typography>
                    <ResponsiveContainer width="100%" height={400}>
                      <PieChart>
                        <Pie
                          data={advancedMetrics.seasonalPattern}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ month, avgPrice }) => `${month}: ₹${avgPrice.toFixed(0)}`}
                          outerRadius={120}
                          fill="#8884d8"
                          dataKey="avgPrice"
                        >
                          {advancedMetrics.seasonalPattern.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => [`₹${value}`, 'Average Price']} />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Market Insights */}
            {marketInsights && (
              <Grid item xs={12}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Analytics color="primary" />
                      Comprehensive Market Insights
                    </Typography>
                    
                    <Grid container spacing={3}>
                      {/* Supply-Demand Factors */}
                      <Grid item xs={12} md={6}>
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMore />}>
                            <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <TrendingUp color="success" />
                              Supply-Demand Factors
                            </Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <List dense>
                              {marketInsights.supplyDemandFactors.map((factor, index) => (
                                <ListItem key={index}>
                                  <ListItemIcon>
                                    <CheckCircle color="success" fontSize="small" />
                                  </ListItemIcon>
                                  <ListItemText primary={factor} />
                                </ListItem>
                              ))}
                            </List>
                          </AccordionDetails>
                        </Accordion>
                      </Grid>

                      {/* External Factors */}
                      <Grid item xs={12} md={6}>
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMore />}>
                            <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Info color="info" />
                              External Market Factors
                            </Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <List dense>
                              {marketInsights.externalFactors.map((factor, index) => (
                                <ListItem key={index}>
                                  <ListItemIcon>
                                    <Warning color="warning" fontSize="small" />
                                  </ListItemIcon>
                                  <ListItemText primary={factor} />
                                </ListItem>
                              ))}
                            </List>
                          </AccordionDetails>
                        </Accordion>
                      </Grid>

                      {/* Recommendations */}
                      <Grid item xs={12}>
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMore />}>
                            <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Assessment color="primary" />
                              Strategic Recommendations
                            </Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <Grid container spacing={2}>
                              {marketInsights.recommendations.map((recommendation, index) => (
                                <Grid item xs={12} md={6} key={index}>
                                  <Card variant="outlined" sx={{ p: 2 }}>
                                    <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <CheckCircle color="success" fontSize="small" />
                                      {recommendation}
                                    </Typography>
                                  </Card>
                                </Grid>
                              ))}
                            </Grid>
                          </AccordionDetails>
                        </Accordion>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Key Metrics Summary */}
            {metrics && (
              <Grid item xs={12}>
                <Card elevation={3}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Key Performance Indicators
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6} sm={3}>
                        <Box textAlign="center" p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="h4" color="primary.main" fontWeight={700}>
                            ₹{metrics.avgPrice}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Average Price
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Box textAlign="center" p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="h4" color="success.main" fontWeight={700}>
                            ₹{metrics.maxPrice}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Highest Price
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Box textAlign="center" p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="h4" color="error.main" fontWeight={700}>
                            ₹{metrics.minPrice}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Lowest Price
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6} sm={3}>
                        <Box textAlign="center" p={2} bgcolor="background.paper" borderRadius={1}>
                          <Typography variant="h4" color="warning.main" fontWeight={700}>
                            {metrics.volatility}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Volatility
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </>
      )}

      {/* Alert System */}
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

export default DetailedAnalysis; 