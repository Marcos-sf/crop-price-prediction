import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { AppBar, Toolbar, Typography, Box, CssBaseline, Container, Button, Avatar } from '@mui/material';
import { Agriculture, TrendingUp } from '@mui/icons-material';
import Dashboard from './pages/Dashboard';
import DetailedAnalysis from './pages/DetailedAnalysis';
import Comparison from './pages/Comparison';
import Export from './pages/Export';
import About from './pages/About';
import News from './pages/News';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Green for agriculture theme
      dark: '#1b5e20',
      light: '#4caf50',
    },
    secondary: {
      main: '#ff9800', // Orange accent
      dark: '#f57c00',
      light: '#ffb74d',
    },
    background: {
      default: '#f8fffe',
      paper: '#ffffff',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

function Navigation() {
  return (
    <AppBar position="static" color="primary" elevation={0} sx={{ 
      background: 'linear-gradient(135deg, #2e7d32 0%, #4caf50 100%)',
      borderBottom: '1px solid rgba(255,255,255,0.1)'
    }}>
      <Container maxWidth="xl">
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Avatar sx={{ 
              bgcolor: 'rgba(255,255,255,0.15)', 
              mr: 2,
              width: 40,
              height: 40
            }}>
              <Agriculture />
            </Avatar>
            <Box>
              <Typography variant="h6" component="div" sx={{ 
                fontWeight: 700,
                fontSize: '1.5rem'
              }}>
                Crop Price Predictor
              </Typography>
              <Typography variant="caption" sx={{ 
                color: 'rgba(255,255,255,0.8)',
                fontSize: '0.75rem'
              }}>
                Smart Price Predictions
              </Typography>
            </Box>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button 
              color="inherit" 
              component={Link} 
              to="/"
              sx={{ 
                mx: 0.5,
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              Dashboard
            </Button>
            <Button 
              color="inherit" 
              component={Link} 
              to="/detailed-analysis"
              sx={{ 
                mx: 0.5,
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              Analysis
            </Button>
            <Button 
              color="inherit" 
              component={Link} 
              to="/comparison"
              sx={{ 
                mx: 0.5,
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              Compare
            </Button>
            <Button 
              color="inherit" 
              component={Link} 
              to="/export"
              sx={{ 
                mx: 0.5,
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              Export
            </Button>
            <Button 
              color="inherit" 
              component={Link} 
              to="/news"
              sx={{ 
                mx: 0.5,
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              News
            </Button>
            <Button 
              color="inherit" 
              component={Link} 
              to="/about"
              sx={{ 
                mx: 0.5,
                borderRadius: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)'
                }
              }}
            >
              About
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navigation />
        <Box 
          minHeight="calc(100vh - 140px)" 
          bgcolor="background.default"
          sx={{
            background: 'linear-gradient(180deg, #f8fffe 0%, #f0f8ff 100%)'
          }}
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/detailed-analysis" element={<DetailedAnalysis />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/export" element={<Export />} />
            <Route path="/about" element={<About />} />
            <Route path="/news" element={<News />} />
          </Routes>
        </Box>
        <Box 
          component="footer" 
          py={3} 
          sx={{
            background: 'linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%)',
            color: 'white',
            textAlign: 'center',
            borderTop: '1px solid rgba(255,255,255,0.1)'
          }}
        >
          <Container maxWidth="lg">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: { xs: 2, md: 0 } }}>
                <TrendingUp sx={{ mr: 1 }} />
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Crop Price Predictor
                </Typography>
              </Box>
              <Box sx={{ textAlign: { xs: 'center', md: 'right' } }}>
                <Typography variant="body2" sx={{ mb: 0.5 }}>
                  Smart Agricultural Price Predictions
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Karnataka, India | Powered by Machine Learning
                </Typography>
              </Box>
            </Box>
          </Container>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
