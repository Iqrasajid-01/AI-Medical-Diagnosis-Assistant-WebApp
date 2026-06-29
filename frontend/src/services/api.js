import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// JWT token interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ── Auth ─────────────────────────────────────
export const login = (email, password) =>
  api.post('/auth/login', { email, password });

export const register = (username, email, password) =>
  api.post('/auth/register', { username, email, password });

export const getProfile = () =>
  api.get('/auth/profile');

// ── Predictions ──────────────────────────────
export const predictDiabetes = (data) =>
  api.post('/predict/diabetes', data);

export const predictHeart = (data) =>
  api.post('/predict/heart', data);

export const predictParkinsons = (formData) =>
  api.post('/predict/parkinsons', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

// ── History ──────────────────────────────────
export const getHistory = (params = {}) =>
  api.get('/predictions/history', { params });

export const getPredictionPdf = (predictionId) =>
  api.get(`/predictions/${predictionId}/pdf`, { responseType: 'blob' });

// ── Admin ────────────────────────────────────
export const adminGetUsers = () =>
  api.get('/admin/users');

export const adminGetPredictions = () =>
  api.get('/admin/predictions');

export const adminRetrain = (diseaseType) =>
  api.post(`/admin/retrain/${diseaseType}`);

export const adminUploadDataset = (diseaseType, formData) =>
  api.post(`/admin/upload-dataset/${diseaseType}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

// ── Doctors ─────────────────────────────────
export const searchNearbyDoctors = (params) =>
  api.post('/doctors/nearby', params);

// ── Appointments ────────────────────────────
export const bookAppointment = (data) =>
  api.post('/appointments/book', data);

export const getAppointments = () =>
  api.get('/appointments');

export const getAppointment = (id) =>
  api.get(`/appointments/${id}`);

export const cancelAppointment = (id) =>
  api.post(`/appointments/${id}/cancel`);

export default api;
