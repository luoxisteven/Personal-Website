// App.js
import React from 'react';
import { SettingsProvider } from './context/SettingsContext';
import AppRoutes from './routes/AppRoutes';

function App() {
  return (
    <SettingsProvider>
      <AppRoutes />
    </SettingsProvider>
  );
}

export default App;
