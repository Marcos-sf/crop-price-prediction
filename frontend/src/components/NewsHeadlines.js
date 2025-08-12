import React, { useEffect, useState } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, Link, Alert, List, ListItem, ListItemText } from '@mui/material';

const NewsHeadlines = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchNews = async () => {
      setLoading(true);
      setError(null);
      try {
        const resp = await fetch('http://localhost:8000/api/v1/news');
        if (!resp.ok) throw new Error('Failed to fetch news');
        const data = await resp.json();
        setNews(data.articles || []);
      } catch (err) {
        setError(err.message || 'Error fetching news');
      }
      setLoading(false);
    };
    fetchNews();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={100}>
        <CircularProgress />
      </Box>
    );
  }
  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }
  if (!news.length) {
    return <Alert severity="info">No news headlines found.</Alert>;
  }

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Latest Crop & Market News
        </Typography>
        <List>
          {news.map((article, idx) => (
            <ListItem key={idx} alignItems="flex-start" divider>
              <ListItemText
                primary={
                  <Link href={article.url} target="_blank" rel="noopener" underline="hover">
                    {article.title}
                  </Link>
                }
                secondary={
                  <>
                    <Typography component="span" variant="body2" color="text.secondary">
                      {article.source} | {new Date(article.publishedAt).toLocaleString()}<br />
                    </Typography>
                    {article.description}
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default NewsHeadlines; 