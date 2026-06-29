import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchNearbyDoctors } from '../../services/api';
import NeonButton from '../UI/NeonButton';

function StarRating({ rating, reviewCount }) {
  if (!rating) return <span className="text-xs text-slate-500">No ratings yet</span>;

  const fullStars = Math.floor(rating);

  return (
    <div className="flex items-center gap-1">
      {[...Array(5)].map((_, i) => (
        <span key={i} className={`text-sm ${i < fullStars ? 'text-yellow-400' : 'text-slate-600'}`}>
          ★
        </span>
      ))}
      <span className="text-xs text-slate-400 ml-1">{rating.toFixed(1)}</span>
      <span className="text-xs text-slate-500">({reviewCount || 0} reviews)</span>
    </div>
  );
}

function BookingButtons({ doctor }) {
  const buttons = [];

  // Website → Book Online
  if (doctor.website) {
    const url = doctor.website.startsWith('http') ? doctor.website : `https://${doctor.website}`;
    buttons.push({
      icon: '🌐',
      text: 'Book Online',
      className: 'bg-gradient-to-r from-green-500 to-emerald-500 hover:opacity-90',
      url,
    });
  }

  // WhatsApp: prefer booking_url wa.me, fallback to phone wa.me (only one WhatsApp)
  const hasWaMeBooking = doctor.booking_url?.includes('wa.me');
  if (hasWaMeBooking) {
    buttons.push({
      icon: '💬',
      text: 'WhatsApp',
      className: 'bg-gradient-to-r from-green-500 to-green-600 hover:opacity-90',
      url: doctor.booking_url,
    });
  } else if (doctor.phone) {
    const cleanPhone = doctor.phone.replace(/\D/g, '');
    buttons.push({
      icon: '💬',
      text: 'WhatsApp',
      className: 'bg-gradient-to-r from-green-500 to-green-600 hover:opacity-90',
      url: `https://wa.me/${cleanPhone}?text=Hi, I would like to book an appointment`,
    });
  }

  // Call Now (only if phone exists)
  if (doctor.phone) {
    buttons.push({
      icon: '📞',
      text: 'Call Now',
      className: 'bg-gradient-to-r from-blue-500 to-blue-600 hover:opacity-90',
      url: `tel:${doctor.phone}`,
    });
  }

  // No website/phone → booking_url as "Book on Google Maps"
  if (!doctor.website && !doctor.phone && doctor.booking_url && !doctor.booking_url.includes('wa.me')) {
    buttons.push({
      icon: '📅',
      text: 'Book on Google Maps',
      className: 'bg-gradient-to-r from-green-500 to-emerald-500 hover:opacity-90',
      url: doctor.booking_url,
    });
  }

  // Get Directions (always)
  buttons.push({
    icon: '📍',
    text: 'Get Directions',
    className: 'bg-gradient-to-r from-primary-500 to-accent-500 hover:opacity-90',
    url: doctor.maps_url,
  });

  return (
    <div className="mt-4 grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(buttons.length, 4)}, 1fr)` }}>
      {buttons.map((btn, idx) => (
        <a
          key={idx}
          href={btn.url}
          target="_blank"
          rel="noopener noreferrer"
          className={`py-2 px-3 rounded-lg text-white text-xs font-medium transition-opacity flex items-center justify-center gap-1.5 ${btn.className}`}
        >
          <span>{btn.icon}</span>
          {btn.text}
        </a>
      ))}
    </div>
  );
}

function DoctorCard({ doctor }) {
  const isOpen = doctor.opening_hours === 'Open Now';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-xl border border-white/10 bg-white/5 hover:border-primary-500/30 transition-all"
    >
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          <h4 className="text-white font-medium">{doctor.name}</h4>
          <p className="text-xs text-primary-400">{doctor.specialty}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          {isOpen ? (
            <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">
              Open
            </span>
          ) : (
            <span className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs rounded-full">
              {doctor.opening_hours || 'Closed'}
            </span>
          )}
        </div>
      </div>

      <StarRating rating={doctor.rating} reviewCount={doctor.review_count} />

      <div className="mt-3 space-y-1.5">
        <p className="text-xs text-slate-400 flex items-center gap-2">
          <span>📍</span>
          <span>{doctor.address}</span>
          <span className="text-primary-400 ml-auto">{doctor.distance_km} km</span>
        </p>

        {doctor.phone && (
          <p className="text-xs text-slate-400 flex items-center gap-2">
            <span>📱</span>
            <a href={`tel:${doctor.phone}`} className="text-primary-400 hover:underline">
              {doctor.phone}
            </a>
          </p>
        )}

        {doctor.website && (
          <p className="text-xs text-slate-400 flex items-center gap-2">
            <span>🌐</span>
            <span className="truncate text-slate-500">{doctor.website.replace('https://', '')}</span>
          </p>
        )}

        {doctor.direct_booking && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">
            ✓ Direct Booking Available
          </span>
        )}
      </div>

      <BookingButtons doctor={doctor} />
    </motion.div>
  );
}

export default function DoctorRecommendation({ result, onClose }) {
  const [location, setLocation] = useState(null);
  const [locationError, setLocationError] = useState('');
  const [loadingLocation, setLoadingLocation] = useState(true);
  const [doctors, setDoctors] = useState([]);
  const [loadingDoctors, setLoadingDoctors] = useState(false);

  const diseaseLabels = {
    diabetes: 'Diabetes',
    heart: 'Heart Disease',
    parkinsons: "Parkinson's Disease"
  };

  useEffect(() => {
    if (!navigator.geolocation) {
      setLocationError('Geolocation not supported by your browser');
      setLoadingLocation(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        });
        setLoadingLocation(false);
      },
      (err) => {
        setLocationError('Location access denied. Please enable location services.');
        setLoadingLocation(false);
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }, []);

  useEffect(() => {
    if (!location) return;

    setLoadingDoctors(true);
    searchNearbyDoctors({
      latitude: location.lat,
      longitude: location.lon,
      disease_type: result.disease,
      risk_level: result.risk_level?.toLowerCase() || 'low',
      limit: 5,
    })
      .then(res => setDoctors(res.data.doctors || []))
      .catch(() => setDoctors([]))
      .finally(() => setLoadingDoctors(false));
  }, [location, result.disease]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-dark-900 rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col"
        >
          <div className="p-6 border-b border-white/10">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <span className="text-2xl">🏥</span>
                  Find Nearby Doctors
                </h2>
                <p className="text-sm text-slate-400 mt-1">
                  Based on your {diseaseLabels[result.disease]} assessment
                </p>
              </div>
              <button onClick={onClose} className="text-slate-400 hover:text-white text-2xl">
                ×
              </button>
            </div>
          </div>

          <div className="p-6 overflow-y-auto flex-1">
            {loadingLocation && (
              <div className="text-center py-12">
                <span className="w-10 h-10 border-3 border-primary-400 border-t-transparent rounded-full animate-spin mx-auto block" />
                <p className="text-slate-400 mt-4">Getting your location...</p>
              </div>
            )}

            {locationError && (
              <div className="text-center py-12">
                <div className="text-5xl mb-4">📍</div>
                <p className="text-slate-300 font-medium">{locationError}</p>
                <p className="text-sm text-slate-500 mt-2 max-w-sm mx-auto">
                  Enable location access to find doctors near you
                </p>
                <button
                  onClick={() => {
                    setLocationError('');
                    setLoadingLocation(true);
                    navigator.geolocation.getCurrentPosition(
                      (pos) => {
                        setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude });
                        setLoadingLocation(false);
                      },
                      () => {
                        setLocationError('Still unable to get location');
                        setLoadingLocation(false);
                      },
                      { enableHighAccuracy: true }
                    );
                  }}
                  className="mt-4 px-4 py-2 rounded-lg bg-primary-500/20 text-primary-400 text-sm hover:bg-primary-500/30 transition-colors"
                >
                  Try Again
                </button>
              </div>
            )}

            {!loadingLocation && !locationError && (
              <>
                <div className="flex items-center gap-2 mb-4 text-sm text-slate-400">
                  <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  Location acquired
                  <span className="text-slate-600">|</span>
                  <span>{location.lat.toFixed(4)}, {location.lon.toFixed(4)}</span>
                </div>

                {loadingDoctors ? (
                  <div className="text-center py-12">
                    <span className="w-10 h-10 border-3 border-primary-400 border-t-transparent rounded-full animate-spin mx-auto block" />
                    <p className="text-slate-400 mt-4">Searching Google Places...</p>
                    <p className="text-xs text-slate-500 mt-1">Finding real doctors near you</p>
                  </div>
                ) : doctors.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="text-5xl mb-4">🔍</div>
                    <p className="text-slate-300 font-medium">No doctors found nearby</p>
                    <p className="text-sm text-slate-500 mt-2">
                      Try searching directly on Google Maps
                    </p>
                    <a
                      href={`https://www.google.com/maps/search/${result.disease}+doctors+near+me`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-500/20 text-primary-400 text-sm hover:bg-primary-500/30 transition-colors"
                    >
                      Open Google Maps
                      <span>↗</span>
                    </a>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between mb-4">
                      <p className="text-sm text-slate-400">
                        {doctors.length} healthcare providers found
                      </p>
                      <div className="flex items-center gap-2 text-xs text-green-400">
                        <span className="w-2 h-2 rounded-full bg-green-400"></span>
                        Real-time data
                      </div>
                    </div>

                    <div className="space-y-3">
                      {doctors.map((doctor, index) => (
                        <DoctorCard key={doctor.id} doctor={doctor} />
                      ))}
                    </div>

                    <div className="mt-6 p-4 bg-white/5 rounded-lg">
                      <p className="text-xs text-slate-500 text-center">
                        💡 Click "Book on Website" or "WhatsApp" to book directly with the doctor's preferred method.
                        Otherwise, click "View on Google Maps" to see all booking options.
                      </p>
                    </div>
                  </>
                )}
              </>
            )}
          </div>

          <div className="p-4 border-t border-white/10">
            <NeonButton onClick={onClose} variant="outline" className="w-full">
              Close
            </NeonButton>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
