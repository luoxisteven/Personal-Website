// Home.js
import React, { useContext } from 'react';
import { Box, Button, Typography } from '@mui/material';
import { SettingsContext } from '../context/SettingsContext';

const Home = () => {
  const { language, themeMode, toggleLanguage, toggleTheme } = useContext(SettingsContext);

  return (
    <Box
      sx={{
        minHeight: '100vh', // 让背景颜色覆盖整个页面
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Typography variant="h4" gutterBottom>
        {language === 'en' ? 'Welcome!' : '欢迎！'}
      </Typography>
      <Button variant="contained" onClick={toggleLanguage} sx={{ mb: 2 }}>
        {language === 'en' ? 'Switch to Chinese' : '切换到英文'}
      </Button>
      <Button variant="outlined" onClick={toggleTheme}>
        Toggle {themeMode === 'light' ? 'Dark Mode' : 'Light Mode'}
      </Button>
    </Box>
  );
};

export default Home;
