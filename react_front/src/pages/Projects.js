import React, { useContext } from 'react';
import { Box, Typography } from '@mui/material';
import CustomAppBar from '../components/AppBar';
import { SettingsContext } from '../context/SettingsContext'; // 引入 SettingsContext

const Projects = () => {
  const { language, themeMode, toggleLanguage, toggleTheme } = useContext(SettingsContext);// 获取当前语言

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <CustomAppBar />
      <Box sx={{ mt: 8, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
          {language === 'en' ? 'Projects!' : '项目！'}
        </Typography>
      </Box>
    </Box>
  );
};

export default Projects;