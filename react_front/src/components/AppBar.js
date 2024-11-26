import React, { useContext, useState } from 'react';
import { alpha, styled } from '@mui/material/styles';
import {
  AppBar,
  Box,
  Toolbar,
  Button,
  IconButton,
  Container,
  Divider,
  MenuItem,
  Drawer,
  Avatar,
} from '@mui/material';
import { Link } from 'react-router-dom';
import MenuIcon from '@mui/icons-material/Menu';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import { SettingsContext } from '../context/SettingsContext';
import ThemeSwitch from './ThemeSwitch';
import LanguageSwitchComponent from './LanguageSwitch';

// 样式化 Toolbar
const StyledToolbar = styled(Toolbar)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  flexShrink: 0,
  borderRadius: `calc(${theme.shape.borderRadius}px + 8px)`,
  backdropFilter: 'blur(24px)',
  border: '1px solid',
  borderColor: (theme.vars || theme).palette.divider,
  backgroundColor: theme.vars
    ? `rgba(${theme.vars.palette.background.defaultChannel} / 0.4)`
    : alpha(theme.palette.background.default, 0.4),
  boxShadow: (theme.vars || theme).shadows[1],
  padding: '8px 12px',
}));

// 按钮的翻译内容
const translations = {
  en: {
    home: 'Home',
    blogs: 'Blogs',
    notes: 'Notes',
    projects: 'Projects',
    finance: 'Finance',
    profile: 'Profile',
    login: 'Log in',
    signup: 'Sign up',
  },
  cn: {
    home: '主页',
    blogs: '博客',
    notes: '笔记',
    projects: '项目',
    finance: '金融面板',
    profile: '个人简介',
    login: '登录',
    signup: '注册',
  },
};

export default function CustomAppBar() {
  const { language, toggleLanguage, themeMode, toggleTheme } = useContext(SettingsContext); // 从 Context 中获取状态和方法
  const [open, setOpen] = useState(false);

  // 切换 Drawer
  const toggleDrawer = (newOpen) => () => {
    setOpen(newOpen);
  };

  return (
    <AppBar
      position="fixed"
      enableColorOnDark
      sx={{
        boxShadow: 0,
        bgcolor: 'transparent',
        backgroundImage: 'none',
        mt: 'calc(var(--template-frame-height, 0px) + 28px)',
      }}
    >
      <Container maxWidth="lg">
        <StyledToolbar variant="dense" disableGutters>
          {/* 圆形头像 */}
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', px: 0 }}>
            <Avatar
              src={require('../assets/img/steven_cartoon_2.JPG')}
              alt="Steven Logo"
              sx={{
                width: 30,
                height: 30,
                mr: 1,
                overflow: 'hidden',
              }}
            />

            {/* 导航按钮 */}
            <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
              <Button component={Link} to="/" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                {translations[language].home}
              </Button>
              <Button component={Link} to="/blogs" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                {translations[language].blogs}
              </Button>
              <Button component={Link} to="/notes" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                {translations[language].notes}
              </Button>
              <Button component={Link} to="/projects" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                {translations[language].projects}
              </Button>
              <Button component={Link} to="/fin" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                {translations[language].finance}
              </Button>
              <Button component={Link} to="/profile" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                {translations[language].profile}
              </Button>
            </Box>
          </Box>

          {/* 右侧按钮 */}
          <Box
            sx={{
              display: { xs: 'none', md: 'flex' },
              gap: 0,
              alignItems: 'center',
            }}
          >
            <LanguageSwitchComponent onClick={toggleLanguage} /> {/* 切换语言 */}
            <ThemeSwitch onClick={toggleTheme} /> {/* 切换主题 */}
          </Box>
          <Box
            sx={{
              display: { xs: 'none', md: 'flex' },
              gap: 1.5,
              alignItems: 'center',
            }}
          >
            <Button component={Link} to="/login" color="primary" variant="outlined" size="small">
              {translations[language].login}
            </Button>
            <Button color="primary" variant="contained" size="small">
              {translations[language].signup}
            </Button>
          </Box>

          {/* 移动端菜单 */}
          <Box sx={{ display: { xs: 'flex', md: 'none' }, gap: 1 }}>
            <IconButton aria-label="Menu button" onClick={toggleDrawer(true)}>
              <MenuIcon />
            </IconButton>
            <Drawer
              anchor="top"
              open={open}
              onClose={toggleDrawer(false)}
              PaperProps={{
                sx: {
                  top: 'var(--template-frame-height, 0px)',
                },
              }}
            >
              <Box sx={{ p: 2, backgroundColor: 'background.default' }}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                  }}
                >
                  <IconButton onClick={toggleDrawer(false)}>
                    <CloseRoundedIcon />
                  </IconButton>
                </Box>
                <MenuItem component={Link} to="/">
                  {translations[language].home}
                </MenuItem>
                <MenuItem component={Link} to="/blogs">
                  {translations[language].blogs}
                </MenuItem>
                <MenuItem component={Link} to="/notes">
                  {translations[language].notes}
                </MenuItem>
                <MenuItem component={Link} to="/projects">
                  {translations[language].projects}
                </MenuItem>
                <MenuItem component={Link} to="/fin">
                  {translations[language].finance}
                </MenuItem>
                <MenuItem component={Link} to="/profile">
                  {translations[language].profile}
                </MenuItem>
                <Divider sx={{ my: 3 }} />
                <MenuItem>
                  <LanguageSwitchComponent onClick={toggleLanguage} />
                </MenuItem>
                <MenuItem>
                  <ThemeSwitch onClick={toggleTheme} />
                </MenuItem>
                <Divider sx={{ my: 3 }} />
                <MenuItem>
                  <Button component={Link} to="/login" color="primary" variant="contained" fullWidth>
                    {translations[language].signup}
                  </Button>
                </MenuItem>
                <MenuItem>
                  <Button component={Link} to="/login" color="primary" variant="outlined" fullWidth>
                    {translations[language].login}
                  </Button>
                </MenuItem>
              </Box>
            </Drawer>
          </Box>
        </StyledToolbar>
      </Container>
    </AppBar>
  );
}
