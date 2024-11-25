// context/SettingsContext.js
import React, { createContext, useState, useMemo } from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';

export const SettingsContext = createContext();

export const SettingsProvider = ({ children }) => {
  const [language, setLanguage] = useState('en'); // 默认语言
  const [themeMode, setThemeMode] = useState('light'); // 默认主题模式

  const toggleLanguage = () => setLanguage(language === 'en' ? 'cn' : 'en');
  const toggleTheme = () => setThemeMode(themeMode === 'light' ? 'dark' : 'light');

  // 动态生成 Material-UI 主题
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: themeMode, // Material 默认支持的 Light/Dark 模式
        },
      }),
    [themeMode]
  );

  return (
    <SettingsContext.Provider value={{ language, themeMode, toggleLanguage, toggleTheme }}>
      <ThemeProvider theme={theme}>
        {/* CssBaseline 应用全局背景颜色和字体颜色 */}
        <CssBaseline />
        {children}
      </ThemeProvider>
    </SettingsContext.Provider>
  );
};
