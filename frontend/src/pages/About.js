import React from 'react';
import { Container, Box, Typography, Paper, Grid, Card, CardContent, Accordion, AccordionSummary, AccordionDetails, List, ListItem, ListItemText, ListItemIcon } from '@mui/material';
import { ExpandMore, Info, Psychology, Analytics, Help } from '@mui/icons-material';

const About = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" align="center" fontWeight={700} gutterBottom>
        About Crop Price Prediction
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" mb={4}>
        A comprehensive tool for predicting crop prices in Karnataka, India
      </Typography>

      <Grid container spacing={3}>
        {/* Project Overview */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Project Overview
              </Typography>
              <Typography variant="body2" paragraph>
                This project aims to predict future prices of major crops in Karnataka using advanced machine learning techniques. 
                The system analyzes historical price data and external factors to provide accurate price forecasts for the next 12 months.
              </Typography>
              <Typography variant="body2" paragraph>
                The predictions help farmers, traders, and policymakers make informed decisions about crop planning, 
                marketing strategies, and agricultural policies.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Supported Crops */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Supported Crops
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Arecanut" secondary="Major commercial crop in coastal Karnataka" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Elaichi (Cardamom)" secondary="High-value spice crop" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Pepper" secondary="Important spice crop" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Coconut" secondary="Major plantation crop" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* How to Use */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                How to Use the Dashboard
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Basic Prediction:
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="1. Select a crop from the dropdown" 
                        secondary="Choose from Arecanut, Elaichi, Pepper, or Coconut"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="2. Select a mandi" 
                        secondary="Choose from 8 major mandis in Karnataka"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="3. Click 'Predict'" 
                        secondary="View historical and predicted prices"
                      />
                    </ListItem>
                  </List>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Advanced Features:
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="• Detailed Analysis" 
                        secondary="In-depth trend and pattern analysis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="• Comparison" 
                        secondary="Compare prices across multiple mandis"
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="• Export Data" 
                        secondary="Download predictions and charts"
                      />
                    </ListItem>
                  </List>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Model Information */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">
                <Psychology sx={{ mr: 1, verticalAlign: 'middle' }} />
                Prediction Models
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    LSTM (Long Short-Term Memory):
                  </Typography>
                  <Typography variant="body2" paragraph>
                    • Captures sequential patterns in time series data
                    • Excellent for long-term dependencies
                    • Handles seasonal variations effectively
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    XGBoost:
                  </Typography>
                  <Typography variant="body2" paragraph>
                    • Handles tabular data with engineered features
                    • Robust to outliers and noise
                    • Can incorporate external factors (weather, news)
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Ensemble Approach:
                  </Typography>
                  <Typography variant="body2">
                    The system combines predictions from both LSTM and XGBoost models to achieve higher accuracy 
                    and more robust forecasts. This ensemble approach reduces the risk of individual model errors 
                    and provides more reliable predictions.
                  </Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Data Sources */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Sources
              </Typography>
              <Typography variant="body2" paragraph>
                • Historical price data from Agmarknet
              </Typography>
              <Typography variant="body2" paragraph>
                • Weather data from meteorological departments
              </Typography>
              <Typography variant="body2" paragraph>
                • News and market sentiment analysis
              </Typography>
              <Typography variant="body2" paragraph>
                • Government agricultural reports
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Contact Information */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Contact & Support
              </Typography>
              <Typography variant="body2" paragraph>
                For technical support or questions about the predictions, please contact:
              </Typography>
              <Typography variant="body2" paragraph>
                Email: support@croppriceprediction.com
              </Typography>
              <Typography variant="body2" paragraph>
                Location: Karnataka, India
              </Typography>
              <Typography variant="body2" paragraph>
                This is a major project developed for agricultural price prediction in Karnataka.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default About; 