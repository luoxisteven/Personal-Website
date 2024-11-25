import React, { useContext } from 'react';
import { styled } from '@mui/material/styles';
import { Switch, FormControlLabel, useMediaQuery } from '@mui/material';
import { SettingsContext } from '../context/SettingsContext';

const LanguageSwitch = styled(Switch)(({ theme }) => ({
  width: 62,
  height: 34,
  padding: 7,
  '& .MuiSwitch-switchBase': {
    // margin: 1,
    padding: 0,
    transform: 'translateX(6px)',
    '&.Mui-checked': {
      color: '#fff',
      transform: 'translateX(22px)',
      '& .MuiSwitch-thumb:before': {
        backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 20 20"><text x="3" y="15" font-size="14" fill="${encodeURIComponent(
          '#fff',
        )}">中</text></svg>')`, // 中文图标
      },
      '& + .MuiSwitch-track': {
        opacity: 1,
        backgroundColor: '#8796A5',
      },
    },
  },
  '& .MuiSwitch-thumb': {
    backgroundColor: '#001e3c',
    width: 32,
    height: 32,
    '&::before': {
      content: "''",
      position: 'absolute',
      width: '100%',
      height: '100%',
      left: 0,
      top: 0,
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'center',
      backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 20 20"><text x="3" y="15" font-size="14" fill="${encodeURIComponent(
        '#fff',
      )}">En</text></svg>')`, // 英文图标
    },
  },
  '& .MuiSwitch-track': {
    opacity: 1,
    backgroundColor: '#aab4be',
    borderRadius: 20 / 2,
  },
}));

export default function LanguageSwitchComponent() {
  const { language, toggleLanguage } = useContext(SettingsContext);
  const isMobile = useMediaQuery((theme) => theme.breakpoints.down('md')); // 检测屏幕宽度

  return (
    <FormControlLabel
      control={
        <LanguageSwitch
          checked={language === 'cn'}
          onChange={toggleLanguage}
        />
      }
      label={isMobile ? (language === 'en' ? 'English' : '中文') : ''} // 移动端显示标签，桌面端隐藏
    />
  );
}
