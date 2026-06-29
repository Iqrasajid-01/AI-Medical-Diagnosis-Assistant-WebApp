import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { getAppointments, cancelAppointment } from '../services/api';
import GlassCard from '../components/UI/GlassCard';
import NeonButton from '../components/UI/NeonButton';

function AppointmentCard({ appointment, onCancel }) {
  const statusColors = {
    confirmed: 'bg-green-500/20 text-green-400 border-green-500/30',
    cancelled: 'bg-red-500/20 text-red-400 border-red-500/30',
    completed: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
  };

  const appointmentDate = appointment.appointment_date?.split('T')[0];
  const isPast = new Date(appointmentDate) < new Date().toISOString().split('T')[0];
  const canCancel = appointment.status === 'confirmed' && !isPast;

  return (
    <GlassCard className="border border-white/10">
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="text-white font-semibold">{appointment.doctor_name}</h3>
          <p className="text-sm text-primary-400">{appointment.doctor_specialty}</p>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs border ${statusColors[appointment.status]}`}>
          {appointment.status.charAt(0).toUpperCase() + appointment.status.slice(1)}
        </span>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <span>📅</span>
          <span>{appointmentDate}</span>
          <span className="text-slate-600">|</span>
          <span>⏰</span>
          <span>{appointment.appointment_time}</span>
        </div>

        {appointment.doctor_address && (
          <div className="flex items-start gap-2 text-sm text-slate-400">
            <span>📍</span>
            <span className="truncate">{appointment.doctor_address}</span>
          </div>
        )}

        {appointment.doctor_phone && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span>📞</span>
            <span>{appointment.doctor_phone}</span>
          </div>
        )}

        {appointment.doctor_rating && (
          <div className="flex items-center gap-2 text-sm">
            <span>⭐</span>
            <span className="text-slate-400">{appointment.doctor_rating.toFixed(1)}</span>
          </div>
        )}

        {appointment.notes && (
          <div className="mt-2 p-2 bg-white/5 rounded text-xs text-slate-400">
            <strong>Notes:</strong> {appointment.notes}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between pt-3 border-t border-white/5">
        <span className="text-xs text-slate-500">
          Booking #{appointment.id}
        </span>

        {canCancel && (
          <NeonButton
            variant="danger"
            size="sm"
            onClick={() => onCancel(appointment.id)}
          >
            Cancel
          </NeonButton>
        )}
      </div>
    </GlassCard>
  );
}

export default function Appointments() {
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    fetchAppointments();
  }, []);

  const fetchAppointments = async () => {
    try {
      const res = await getAppointments();
      setAppointments(res.data.appointments || []);
    } catch (err) {
      setError('Failed to load appointments');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (id) => {
    if (!confirm('Cancel this appointment?')) return;

    try {
      await cancelAppointment(id);
      setAppointments(prev =>
        prev.map(apt => apt.id === id ? { ...apt, status: 'cancelled' } : apt)
      );
    } catch (err) {
      alert(err.response?.data?.error || 'Failed to cancel');
    }
  };

  const filteredAppointments = appointments.filter(apt => {
    if (filter === 'all') return true;
    if (filter === 'upcoming') {
      const aptDate = new Date(apt.appointment_date).toISOString().split('T')[0];
      const today = new Date().toISOString().split('T')[0];
      return aptDate >= today && apt.status === 'confirmed';
    }
    if (filter === 'past') {
      const aptDate = new Date(apt.appointment_date).toISOString().split('T')[0];
      const today = new Date().toISOString().split('T')[0];
      return aptDate < today;
    }
    return apt.status === filter;
  });

  const statusCounts = {
    all: appointments.length,
    upcoming: appointments.filter(a => {
      const aptDate = new Date(a.appointment_date).toISOString().split('T')[0];
      return aptDate >= new Date().toISOString().split('T')[0] && a.status === 'confirmed';
    }).length,
    cancelled: appointments.filter(a => a.status === 'cancelled').length,
  };

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-white mb-2">My Appointments</h1>
        <p className="text-slate-400">View and manage your booked appointments</p>
      </motion.div>

      <div className="flex gap-2">
        {['all', 'upcoming', 'cancelled'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              filter === f
                ? 'bg-primary-500/20 text-primary-400'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
            <span className="ml-2 text-xs opacity-60">({statusCounts[f]})</span>
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <span className="w-8 h-8 border-2 border-primary-400 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : error ? (
        <GlassCard className="text-center py-8">
          <p className="text-danger-400">{error}</p>
          <NeonButton variant="outline" onClick={fetchAppointments} className="mt-4">
            Retry
          </NeonButton>
        </GlassCard>
      ) : filteredAppointments.length === 0 ? (
        <GlassCard className="text-center py-12">
          <div className="text-5xl mb-4">📅</div>
          <h3 className="text-lg text-white mb-2">No appointments</h3>
          <p className="text-slate-400 text-sm">
            {filter === 'upcoming'
              ? 'Book an appointment from your prediction results'
              : 'No appointments found'}
          </p>
        </GlassCard>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {filteredAppointments.map(apt => (
            <AppointmentCard
              key={apt.id}
              appointment={apt}
              onCancel={handleCancel}
            />
          ))}
        </div>
      )}
    </div>
  );
}
