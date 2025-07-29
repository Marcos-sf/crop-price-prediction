import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { AppBar, Toolbar, Typography, Box, CssBaseline, Container, Button } from '@mui/material';
import Dashboard from './pages/Dashboard';
import DetailedAnalysis from './pages/DetailedAnalysis';
import Comparison from './pages/Comparison';
import Export from './pages/Export';
import About from './pages/About';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f4f6f8',
    },
  },
});

function Navigation() {
  return (
    <AppBar position="static" color="primary" elevation={2}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Crop Price Prediction
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          <Button color="inherit" component={Link} to="/detailed-analysis">
            Detailed Analysis
          </Button>
          <Button color="inherit" component={Link} to="/comparison">
            Comparison
          </Button>
          <Button color="inherit" component={Link} to="/export">
            Export
          </Button>
          <Button color="inherit" component={Link} to="/about">
            About
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navigation />
        <Box minHeight="90vh" bgcolor="background.default">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/detailed-analysis" element={<DetailedAnalysis />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/export" element={<Export />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Box>
        <Box component="footer" py={2} bgcolor="primary.main" color="white" textAlign="center">
          <Container maxWidth="md">
            <Typography variant="body2">
              Crop Price Prediction | Karnataka, India
            </Typography>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
