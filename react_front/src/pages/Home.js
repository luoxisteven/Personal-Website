import React, { useContext } from 'react';
import { Box, Container } from '@mui/material';
import CustomAppBar from '../components/AppBar';
import ProfileCard from "../components/ProfileCard";
import { SettingsContext } from '../context/SettingsContext';
import MarkdownViewer from "../components/MarkdownViewer";

const Home = () => {
  const { language, themeMode, toggleLanguage, toggleTheme } = useContext(SettingsContext);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* AppBar 部分 */}
      <CustomAppBar />

      {/* 主内容部分 */}
      <Container
        maxWidth="lg"
        sx={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center', // 保证水平方向对齐
          justifyContent: 'space-between', // 水平方向分布空间
          paddingTop: '64px', // 避免与 AppBar 重叠
          position: 'relative',
          height: 'calc(100vh - 64px)', // 去掉 AppBar 的高度
        }}
      >
        {/* ProfileCard 部分 */}
        <Box
          sx={{
            flex: '1 1 20%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%', // 保持 ProfileCard 容器高度为 100%
            marginRight: '0px',
            margin: 0, // 确保没有额外间距
            padding: 0, // 确保内部没有额外间距
          }}
        >
          <ProfileCard />
        </Box>

        {/* MarkdownViewer 部分 */}
        <Box
          sx={{
            flex: '1 1 80%',
            height: '100%',
            overflowY: 'auto', // 启用垂直滚动
            borderRadius: '8px', // 添加圆角
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)', // 添加阴影效果
            margin: 0, // 确保没有额外间距
            padding: 0, // 确保内部没有额外间距
          }}
        >
          <MarkdownViewer filePath="md/luoxisteven.md" />
        </Box>
      </Container>
    </Box>
  );
};

export default Home;
