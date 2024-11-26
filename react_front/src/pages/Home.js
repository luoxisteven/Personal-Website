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
        marginTop: '64px', // 避免与 AppBar 重叠
        height: 'calc(100vh - 64px)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* AppBar 部分 */}
      <CustomAppBar />

      {/* 主内容部分 - 桌面端 */}
      <Container
        maxWidth="lg"
        sx={{
          display: { xs: 'none', md: 'flex' }, // 仅桌面端显示
          alignItems: 'center', // 确保子元素在 Y 轴居中
          justifyContent: 'space-between',
          height: 'calc(100vh - 64px)', // 减去 AppBar 的高度
        }}
      >
        {/* ProfileCard 部分 */}
        <Box
          sx={{
            flex: '1 1 20%',
            display: 'flex',
            alignItems: 'center', // 子项 Y 轴居中
          }}
        >
          <ProfileCard />
        </Box>

        {/* MarkdownViewer 部分 */}
        <Box
          sx={{
            flex: '1 1 80%',
            height: '100%', // 高度撑满父级
            display: 'flex',
            alignItems: 'center', // 子项 Y 轴居中
          }}
        >
          <MarkdownViewer filePath="md/luoxisteven.md" />
        </Box>
      </Container>

      {/* 主内容部分 - 移动端 */}
      <Container
        maxWidth="sm"
        sx={{
          display: { xs: 'flex', md: 'none' }, // 仅移动端显示
          flexDirection: 'column', // 垂直堆叠
          alignItems: 'center', // 确保子元素在 X 轴居中
          justifyContent: 'flex-start', // 从顶部开始布局
          height: 'calc(100vh - 64px)', // 减去 AppBar 的高度
          gap: 2, // 子项间距
          marginTop: '64px', // 为移动端内容添加顶部间距
        }}
      >
        {/* ProfileCard 部分 */}
        <Box
          sx={{
            width: '100%', // 占满宽度
            display: 'flex',
            justifyContent: 'center', // 居中对齐
          }}
        >
          <ProfileCard />
        </Box>

        {/* MarkdownViewer 部分 */}
        <Box
          sx={{
            width: '100%', // 占满宽度
            display: 'flex',
            justifyContent: 'center', // 居中对齐
          }}
        >
          <MarkdownViewer filePath="md/luoxisteven.md" />
        </Box>
      </Container>
    </Box>
  );
};

export default Home;
