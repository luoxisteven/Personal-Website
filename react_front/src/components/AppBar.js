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
import { Link } from 'react-router-dom'; // 引入 Link
import MenuIcon from '@mui/icons-material/Menu';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import ThemeSwitch from './ThemeSwitch';
import LanguageSwitchComponent from './LanguageSwitch';

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

export default function CustomAppBar() {
  const [open, setOpen] = useState(false);

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
              src={require('../assets/img/steven_cartoon.JPG')}
              alt="Steven Logo"
              sx={{
                width: 30,
                height: 30,
                mr: 1,
                overflow: 'hidden',
              }}
            />

            <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
              <Button component={Link} to="/" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                Home
              </Button>
              <Button component={Link} to="/blogs" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                Blogs
              </Button>
              <Button component={Link} to="/projects" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                Projects
              </Button>
              <Button component={Link} to="/fin" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                Finance
              </Button>
              <Button component={Link} to="/profile" variant="text" color="info" size="small" sx={{ minWidth: 0 }}>
                Profile
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
            <LanguageSwitchComponent /> {/* 使用语言切换组件 */}
            <ThemeSwitch /> {/* 使用主题切换组件 */}
          </Box>
          <Box
            sx={{
              display: { xs: 'none', md: 'flex' },
              gap: 1.5,
              alignItems: 'center',
            }}
          >
            <Button component={Link} to="/login" color="primary" variant="outlined" size="small">
              Log in
            </Button>
            <Button color="primary" variant="contained" size="small">
              Sign up
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
                  Home
                </MenuItem>
                <MenuItem component={Link} to="/blogs">
                  Blogs
                </MenuItem>
                <MenuItem component={Link} to="/projects">
                  Projects
                </MenuItem>
                <MenuItem component={Link} to="/fin">
                  Finance
                </MenuItem>
                <MenuItem component={Link} to="/profile">
                  Profile
                </MenuItem>
                <Divider sx={{ my: 3 }} />
                <MenuItem>
                  <LanguageSwitchComponent /> {/* 在移动端菜单中使用语言切换组件 */}
                </MenuItem>
                <MenuItem>
                  <ThemeSwitch /> {/* 在移动端菜单中使用主题切换组件 */}
                </MenuItem>
                <Divider sx={{ my: 3 }} />
                <MenuItem>
                  <Button component={Link} to="/login" color="primary" variant="contained" fullWidth>
                    Sign up
                  </Button>
                </MenuItem>
                <MenuItem>
                  <Button component={Link} to="/login" color="primary" variant="outlined" fullWidth>
                    Sign in
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
