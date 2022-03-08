import React, { useEffect, useState } from 'react';
import { IconButton } from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import { getValueFromLocalStorage, setValueToLocalStorage } from '../../utils/localStorage';
import './style.scss';

function ThemeSwitch() {
  const [isLightMode, setIsLightMode] = useState(getValueFromLocalStorage('isLightMode'));

  useEffect(() => {
    setValueToLocalStorage('isLightMode', isLightMode);
    document.documentElement.setAttribute('data-theme', isLightMode ? 'light' : 'dark');
  }, [isLightMode]);

  return (
    <div className="dark-mode-button-wrapper">
      <IconButton className="dark-mode-button" onClick={() => setIsLightMode((isLight) => !isLight)}>
        {isLightMode ? (
          <DarkModeIcon className="dark-mode-icon" fontSize="large" />
        ) : (
          <LightModeIcon className="dark-mode-icon" fontSize="large" />
        )}
      </IconButton>
    </div>
  );
}

export default ThemeSwitch;