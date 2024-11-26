import React, { createContext, useState, useMemo, useEffect } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';

// 创建 Settings 上下文
export const SettingsContext = createContext();

export const SettingsProvider = ({ children }) => {
  // 语言状态
  const [language, setLanguage] = useState('en'); // 默认语言为英文
  // 主题模式状态
  const [themeMode, setThemeMode] = useState('light'); // 默认主题为浅色模式

  // 检测系统语言
  useEffect(() => {
    const systemLanguage = navigator.language.startsWith('zh') ? 'cn' : 'en'; // 检测是否是中文环境
    setLanguage(systemLanguage);
  }, []);

  // 检测系统主题
  useEffect(() => {
    const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)'); // 检测系统是否为暗色模式
    const systemThemeMode = darkModeMediaQuery.matches ? 'dark' : 'light';
    setThemeMode(systemThemeMode);

    // 监听系统主题变化
    const handleChange = (e) => {
      setThemeMode(e.matches ? 'dark' : 'light');
    };
    darkModeMediaQuery.addEventListener('change', handleChange);

    // 清除事件监听
    return () => {
      darkModeMediaQuery.removeEventListener('change', handleChange);
    };
  }, []);

  // 切换语言
  const toggleLanguage = () => setLanguage(language === 'en' ? 'cn' : 'en');
  // 切换主题模式
  const toggleTheme = () => setThemeMode(themeMode === 'light' ? 'dark' : 'light');

  // 动态创建 Material-UI 主题
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: themeMode, // 应用当前主题模式
        },
      }),
    [themeMode]
  );

  return (
    <SettingsContext.Provider value={{ language, themeMode, toggleLanguage, toggleTheme }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </SettingsContext.Provider>
  );
};
