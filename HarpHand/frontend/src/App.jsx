import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Login from './pages/Login';
import Tool from './pages/Tool';
import { ThemeProvider } from './components/ThemeToggle';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/tool" element={<Tool />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
