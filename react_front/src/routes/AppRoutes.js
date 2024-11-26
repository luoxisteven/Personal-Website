// AppRoutes.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from '../pages/Home';
import Blogs from '../pages/Blogs';
import Notes from '../pages/Notes';
import Finance from '../pages/Finance';
import Login from '../pages/Login';
import Profile from '../pages/Profile'
import Projects from '../pages/Projects'

const AppRoutes = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/blogs" element={<Blogs />} />
        <Route path="/notes" element={<Notes />} />
        <Route path="/fin" element={<Finance />} />
        <Route path="/login" element={<Login />} />
        <Route path="/profile" element={<Profile/>} />
        <Route path="/projects" element={<Projects/>} />
      </Routes>
    </Router>
  );
};

export default AppRoutes;
