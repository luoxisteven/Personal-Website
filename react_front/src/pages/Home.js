import React, { useContext } from 'react';
import { Box, Container } from '@mui/material';
import CustomAppBar from '../components/AppBar';
import ProfileCard from "../components/ProfileCard";
import { SettingsContext } from '../context/SettingsContext';
import MarkdownViewer from "../components/MarkdownViewer";

const Home = () => {
  const { language, themeMode, toggleLanguage, toggleTheme } = useContext(SettingsContext); // 获取当前语言

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
      }}
    >
      {/* AppBar 部分 */}
      <CustomAppBar />

      {/* ProfileCard 部分 */}
      <Container
        maxWidth="lg"
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-start', // 将内容左对齐
          minHeight: 'calc(100vh - 64px)', // 让内容垂直居中，减去 AppBar 的高度
        }}
      >
        <Box sx={{ textAlign: 'left', marginLeft: '50px' }}> {/* 确保文字内容左对齐 */}
          <ProfileCard />
        </Box>
        <MarkdownViewer filePath="md/luoxisteven.md" />
      </Container>
    </Box>
  );
};

export default Home;
