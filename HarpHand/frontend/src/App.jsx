import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import Tool from './pages/Tool';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/tool" element={<Tool />} />
      </Routes>
    </BrowserRouter>
  );
}
