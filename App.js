import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const EnhancedShipOptimizer = () => {
  const [optimizationData, setOptimizationData] = useState({
    departure_port: "Khalifa Port",
    arrival_port: "Ruwais Port", 
    required_arrival_time: "",
    wind_speed: 5.0,
    wind_direction: 90.0
  });

  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [weatherData, setWeatherData] = useState(null);
  const [activeTab, setActiveTab] = useState("optimization");

  // Set default arrival time (24 hours from now)
  useEffect(() => {
    const now = new Date();
    now.setHours(now.getHours() + 24);
    now.setMinutes(0);
    now.setSeconds(0);
    const isoString = now.toISOString().slice(0, 16);
    setOptimizationData(prev => ({
      ...prev,
      required_arrival_time: isoString
    }));
  }, []);

  // Fetch weather data
  useEffect(() => {
    fetchWeatherData();
  }, [optimizationData.departure_port, optimizationData.arrival_port]);

  const fetchWeatherData = async () => {
    try {
      const response = await axios.get(`${API}/weather`, {
        params: {
          departure_port: optimizationData.departure_port,
          arrival_port: optimizationData.arrival_port
        }
      });
      setWeatherData(response.data);
      
      // Update wind inputs with real data
      if (response.data.average) {
        setOptimizationData(prev => ({
          ...prev,
          wind_speed: response.data.average.wind_speed,
          wind_direction: response.data.average.wind_direction
        }));
      }
    } catch (error) {
      console.error("Error fetching weather data:", error);
    }
  };

  const handleOptimize = async () => {
    setIsLoading(true);
    setError("");
    setResults(null);

    // Check if departure and arrival ports are the same
    if (optimizationData.departure_port === optimizationData.arrival_port) {
      setError("Departure port and arrival port cannot be the same. Please select different ports.");
      setIsLoading(false);
      return;
    }

    try {
      const response = await axios.post(`${API}/optimize`, {
        ...optimizationData,
        priority_weights: { fuel: 0.6, time: 0.4 }, // Default balanced weights
        request_alternatives: true
      });

      setResults(response.data);
      setActiveTab("results");
    } catch (error) {
      setError(error.response?.data?.detail || "Optimization failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setOptimizationData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100">
      {/* Header */}
      <header className="bg-blue-900 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">🚢 M/V Al-bazm II Enhanced Maritime Optimizer</h1>
          <p className="text-blue-200 mt-2">Advanced Multi-Objective Route & Fuel Optimization System</p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6">
              {[
                { id: "optimization", label: "Route Optimization", icon: "🧭" },
                { id: "results", label: "Results & Analysis", icon: "📊" },
                { id: "weather", label: "Weather Conditions", icon: "🌊" }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? "border-blue-500 text-blue-600"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  {tab.icon} {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Optimization Tab */}
        {activeTab === "optimization" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Voyage Configuration */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">🛳️ Voyage Planning</h2>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Departure Port
                    </label>
                    <select
                      value={optimizationData.departure_port}
                      onChange={(e) => handleInputChange("departure_port", e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="Khalifa Port">Khalifa Port</option>
                      <option value="Ruwais Port">Ruwais Port</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Arrival Port
                    </label>
                    <select
                      value={optimizationData.arrival_port}
                      onChange={(e) => handleInputChange("arrival_port", e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="Ruwais Port">Ruwais Port</option>
                      <option value="Khalifa Port">Khalifa Port</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Required Arrival Time (Dubai Time)
                  </label>
                  <input
                    type="datetime-local"
                    value={optimizationData.required_arrival_time}
                    onChange={(e) => handleInputChange("required_arrival_time", e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Wind Speed (m/s)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={optimizationData.wind_speed}
                      onChange={(e) => handleInputChange("wind_speed", parseFloat(e.target.value))}
                      className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Wind Direction (°)
                    </label>
                    <input
                      type="number"
                      step="1"
                      value={optimizationData.wind_direction}
                      onChange={(e) => handleInputChange("wind_direction", parseFloat(e.target.value))}
                      className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>

                <div className="mt-6">
                  <button
                    onClick={handleOptimize}
                    disabled={isLoading}
                    className={`w-full py-3 px-6 rounded-md font-medium text-white transition-colors ${
                      isLoading
                        ? "bg-gray-400 cursor-not-allowed"
                        : "bg-blue-600 hover:bg-blue-700 focus:ring-2 focus:ring-blue-500"
                    }`}
                  >
                    {isLoading ? "🔄 Optimizing..." : "🚀 Optimize Route"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === "results" && results && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">📊 Recommended Route: {results.recommended_route.route_name}</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <MetricCard
                  title="Total Distance"
                  value={results.recommended_route.total_distance_nm}
                  unit="nm"
                  icon="📏"
                />
                <MetricCard
                  title="Total Trip Time"
                  value={results.recommended_route.estimated_duration_hours}
                  unit="hours"
                  icon="⏱️"
                />
                <MetricCard
                  title="Estimated Fuel"
                  value={results.recommended_route.total_fuel_mt}
                  unit="MT"
                  icon="⛽"
                />
                <MetricCard
                  title="Optimization Score"
                  value={results.recommended_route.optimization_score}
                  unit=""
                  icon="🎯"
                />
              </div>
            </div>

            {/* Route Visualization */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">🗺️ Route Visualization</h3>
              <RouteMap waypoints={results.recommended_route.waypoints} />
            </div>

            {/* Route Details Table */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">📋 Waypoint Details</h3>
              <RouteTable waypoints={results.recommended_route.waypoints} />
            </div>

            {/* Optimization Insights */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">💡 Optimization Insights</h3>
              <div className="space-y-3">
                {Object.entries(results.optimization_insights)
                  .filter(([key]) => key !== 'recommendation_rationale') // Filter out recommendation rationale
                  .map(([key, value]) => (
                  <div key={key} className="p-3 bg-blue-50 rounded-md">
                    <h4 className="font-medium text-blue-900 capitalize">{key.replace('_', ' ')}</h4>
                    <p className="text-blue-800 mt-1">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Weather Tab */}
        {activeTab === "weather" && weatherData && (
          <div className="space-y-6">
            <WeatherDisplay weatherData={weatherData} />
          </div>
        )}
      </div>
    </div>
  );
};

// Metric Card Component
const MetricCard = ({ title, value, unit, icon }) => (
  <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-gray-600">{title}</p>
        <p className="text-2xl font-bold text-blue-900">{value}</p>
        {unit && <p className="text-sm text-gray-500">{unit}</p>}
      </div>
      <div className="text-2xl">{icon}</div>
    </div>
  </div>
);

// Enhanced Route Map Component
const RouteMap = ({ waypoints }) => {
  useEffect(() => {
    if (!window.L || !waypoints?.length) return;

    // Initialize map
    const map = window.L.map('route-map').setView([24.5, 53.7], 9);
    
    // Add tile layer
    window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Create route line
    const routeCoordinates = waypoints.map(wp => [wp.lat, wp.lon]);
    const routeLine = window.L.polyline(routeCoordinates, {
      color: '#2563eb',
      weight: 4,
      opacity: 0.8
    }).addTo(map);

    // Add waypoint markers
    waypoints.forEach((waypoint, index) => {
      const isEndpoint = index === 0 || index === waypoints.length - 1;
      
      const icon = window.L.divIcon({
        className: 'custom-marker',
        html: `<div class="marker ${isEndpoint ? 'endpoint' : 'waypoint'}">
                 <span class="marker-icon">${isEndpoint ? '⚓' : '📍'}</span>
               </div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 30]
      });

      const marker = window.L.marker([waypoint.lat, waypoint.lon], { icon }).addTo(map);
      
      marker.bindPopup(`
        <div class="p-2">
          <h4 class="font-bold">${waypoint.name}</h4>
          <p>Lat: ${waypoint.lat.toFixed(4)}</p>
          <p>Lon: ${waypoint.lon.toFixed(4)}</p>
          ${waypoint.suggested_speed_kn ? `<p>Speed: ${waypoint.suggested_speed_kn} knots</p>` : ''}
          ${waypoint.course_to_next ? `<p>Course: ${waypoint.course_to_next}°</p>` : ''}
        </div>
      `);
    });

    // Fit map to route bounds
    map.fitBounds(routeLine.getBounds(), { padding: [20, 20] });

    // Cleanup function
    return () => {
      map.remove();
    };
  }, [waypoints]);

  return (
    <div className="space-y-4">
      <div id="route-map" className="h-96 w-full rounded-lg border border-gray-300"></div>
      <style jsx>{`
        .custom-marker .marker {
          background: white;
          border-radius: 50%;
          border: 3px solid #2563eb;
          display: flex;
          align-items: center;
          justify-content: center;
          width: 30px;
          height: 30px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .custom-marker .endpoint {
          border-color: #dc2626;
          background: #fef2f2;
        }
        .custom-marker .marker-icon {
          font-size: 16px;
        }
      `}</style>
    </div>
  );
};

// Route Table Component
const RouteTable = ({ waypoints }) => (
  <div className="overflow-x-auto">
    <table className="min-w-full divide-y divide-gray-200">
      <thead className="bg-gray-50">
        <tr>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
            Waypoint
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
            Coordinates
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
            Course (°)
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
            Distance (nm)
          </th>
          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
            Speed (kn)
          </th>
        </tr>
      </thead>
      <tbody className="bg-white divide-y divide-gray-200">
        {waypoints.map((waypoint, index) => (
          <tr key={index} className="hover:bg-gray-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
              {waypoint.name}
            </td>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
              {waypoint.lat.toFixed(4)}, {waypoint.lon.toFixed(4)}
            </td>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
              {waypoint.course_to_next || 'N/A'}
            </td>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
              {waypoint.distance_to_next_nm?.toFixed(1) || 'N/A'}
            </td>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
              {waypoint.suggested_speed_kn}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

// Weather Display Component
const WeatherDisplay = ({ weatherData }) => {
  const getLocationName = (location, locationData) => {
    if (location === 'departure') {
      return locationData?.location_name || 'Departure Port';
    } else if (location === 'arrival') {
      return locationData?.location_name || 'Arrival Port';
    } else {
      return 'Route Midpoint';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {['departure', 'midpoint', 'arrival'].map((location) => (
        <div key={location} className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-800 capitalize">
            {getLocationName(location, weatherData[location])}
          </h3>
          
          {weatherData[location] && (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Temperature:</span>
                <span className="font-medium">{weatherData[location].temperature?.toFixed(1)}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Wind Speed:</span>
                <span className="font-medium">{weatherData[location].wind_speed?.toFixed(1)} m/s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Wind Direction:</span>
                <span className="font-medium">{weatherData[location].wind_direction?.toFixed(0)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Sea State:</span>
                <span className="font-medium">Level {weatherData[location].sea_state || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Wave Height:</span>
                <span className="font-medium">{weatherData[location].wave_height?.toFixed(1)}m</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Impact Score:</span>
                <span className={`font-medium ${
                  weatherData[location].impact_score > 1.1 ? 'text-red-600' : 
                  weatherData[location].impact_score < 0.9 ? 'text-green-600' : 'text-yellow-600'
                }`}>
                  {weatherData[location].impact_score?.toFixed(2)}
                </span>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// Route Comparison Component
const RouteComparison = ({ recommendedRoute, alternativeRoutes, comparisonMatrix }) => (
  <div className="space-y-6">
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">📊 Route Comparison Analysis</h2>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Route
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Type
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Fuel (MT)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Duration (h)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Score
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {/* Recommended route */}
            <tr className="bg-green-50">
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-green-900">
                ⭐ {recommendedRoute.route_name}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {recommendedRoute.route_type}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {recommendedRoute.total_fuel_mt.toFixed(2)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {recommendedRoute.estimated_duration_hours.toFixed(1)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600 font-medium">
                {recommendedRoute.optimization_score.toFixed(3)}
              </td>
            </tr>
            
            {/* Alternative routes */}
            {alternativeRoutes.map((route, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {route.route_name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {route.route_type}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {route.total_fuel_mt.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {route.estimated_duration_hours.toFixed(1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {route.optimization_score.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>

    {/* Best in Category */}
    {comparisonMatrix && (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="font-semibold text-green-800">🏆 Best Fuel Efficiency</h3>
          <p className="text-sm text-gray-600">{comparisonMatrix.best_fuel?.route}</p>
          <p className="text-lg font-bold text-green-600">{comparisonMatrix.best_fuel?.value?.toFixed(2)} MT</p>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="font-semibold text-blue-800">⚡ Fastest Route</h3>
          <p className="text-sm text-gray-600">{comparisonMatrix.best_time?.route}</p>
          <p className="text-lg font-bold text-blue-600">{comparisonMatrix.best_time?.value?.toFixed(1)} hours</p>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="font-semibold text-orange-800">🎯 Best Overall</h3>
          <p className="text-sm text-gray-600">{comparisonMatrix.best_overall?.route}</p>
          <p className="text-lg font-bold text-orange-600">{comparisonMatrix.best_overall?.value?.toFixed(3)}</p>
        </div>
      </div>
    )}
  </div>
);

export default EnhancedShipOptimizer;