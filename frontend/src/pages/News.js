import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import NewsHeadlines from '../components/NewsHeadlines';

const News = () => {
  return (
    <Container maxWidth="md">
      <Box my={4}>
        <Typography variant="h4" gutterBottom>
          Crop & Market News
        </Typography>
        <NewsHeadlines />
      </Box>
    </Container>
  );
};

export default News; 