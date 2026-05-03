import 'package:flutter/material.dart';

import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';

import '../../models/soil_reading.dart';
import '../../services/soil_firestore_service.dart';
import '../../services/location_service.dart';
import '../../services/open_meteo_service.dart';

class SoilQualityScreen extends StatefulWidget {
  const SoilQualityScreen({super.key});

  @override
  State<SoilQualityScreen> createState() => _SoilQualityScreenState();
}

class _SoilQualityScreenState extends State<SoilQualityScreen> {
  final _soilService = SoilFirestoreService();

  bool _loadingEnv = true;
  String _location = "Locating...";
  OpenMeteoWeather? _weather;
  String? _envError;

  @override
  void initState() {
    super.initState();
    _loadEnv();
  }

  Future<void> _loadEnv() async {
    setState(() {
      _loadingEnv = true;
      _envError = null;
    });

    try {
      final loc = await LocationService.getCurrentLocationLabel();
      final w = await OpenMeteoService.current(lat: loc.lat, lon: loc.lon);

      if (!mounted) return;
      setState(() {
        _location = loc.label;
        _weather = w;
        _loadingEnv = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _envError = e.toString();
        _loadingEnv = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===== MODERN HEADER =====
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
            decoration: const BoxDecoration(
              gradient: AppColors.heroGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(28),
                bottomRight: Radius.circular(28),
              ),
            ),
            child: Stack(
              children: [
                const Positioned(
                  right: -18,
                  top: 18,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.terrain_rounded,
                        size: 150, color: Colors.white),
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        IconButton(
                          onPressed: () => Navigator.of(context).pop(),
                          icon: const Icon(Icons.arrow_back_ios_new_rounded),
                          color: Colors.white,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          "Soil Quality",
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.92),
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                        const Spacer(),
                        InkWell(
                          borderRadius: BorderRadius.circular(999),
                          onTap: _loadEnv,
                          child: CircleAvatar(
                            radius: 18,
                            backgroundColor: Colors.white.withOpacity(0.12),
                            child: const Icon(Icons.refresh_rounded,
                                color: Colors.white),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Live Soil & Environment",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.96),
                        fontWeight: FontWeight.w900,
                        fontSize: 22,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      "Realtime soil readings from ESP + weather snapshot",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontWeight: FontWeight.w600,
                        fontSize: 12.8,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // Location pill
                    _headerPill(
                      icon: Icons.location_on_rounded,
                      text: _location,
                    ),
                    const SizedBox(height: 12),

                    // Banner with overlay
                    ClipRRect(
                      borderRadius: BorderRadius.circular(18),
                      child: SizedBox(
                        height: 130,
                        width: double.infinity,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.asset(
                              "assets/images/soilbanner.jpg",
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => Container(
                                color: Colors.white.withOpacity(0.10),
                                child: const Center(
                                  child: Icon(Icons.terrain_rounded,
                                      color: AppColors.primary, size: 44),
                                ),
                              ),
                            ),
                            Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  begin: Alignment.topCenter,
                                  end: Alignment.bottomCenter,
                                  colors: [
                                    Colors.black.withOpacity(0.05),
                                    Colors.black.withOpacity(0.45),
                                  ],
                                ),
                              ),
                            ),
                            Positioned(
                              left: 12,
                              bottom: 12,
                              right: 12,
                              child: Row(
                                children: [
                                  _statusChip(
                                    _loadingEnv
                                        ? "Updating..."
                                        : (_envError != null
                                            ? "Weather error"
                                            : "Live"),
                                    color: _envError != null
                                        ? const Color(0xFFE05C5C)
                                        : const Color(0xFF2FA36B),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Text(
                                      _loadingEnv
                                          ? "Fetching weather..."
                                          : (_envError != null
                                              ? "Tap refresh to try again"
                                              : "${_weather?.weatherLabel ?? "—"} • wind ${(_weather?.windSpeed ?? 0).toStringAsFixed(0)} km/h"),
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // ===== BODY =====
          Expanded(
            child: RefreshIndicator(
              onRefresh: _loadEnv,
              child: StreamBuilder<SoilReading?>(
                stream: _soilService.latestSoilStream(),
                builder: (context, snap) {
                  final soil = snap.data;

                  return ListView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
                    children: [
                      // ===== ENV SNAPSHOT (modern row) =====
                      _whiteCard(
                        child: Row(
                          children: [
                            Container(
                              width: 44,
                              height: 44,
                              decoration: BoxDecoration(
                                color: AppColors.primary.withOpacity(0.18),
                                borderRadius: BorderRadius.circular(14),
                              ),
                              child: const Icon(Icons.cloud_rounded,
                                  color: AppColors.darkGreen),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                "Environment Snapshot",
                                style: TextStyle(
                                  color: AppColors.textDark.withOpacity(0.88),
                                  fontWeight: FontWeight.w900,
                                  fontSize: 14.5,
                                ),
                              ),
                            ),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: AppColors.darkGreen.withOpacity(0.06),
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(color: AppColors.border),
                              ),
                              child: Text(
                                "FREE API",
                                style: TextStyle(
                                  color: AppColors.darkGreen.withOpacity(0.85),
                                  fontWeight: FontWeight.w900,
                                  fontSize: 11.5,
                                  letterSpacing: 0.8,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 12),

                      if (_envError != null)
                        _whiteCard(
                          child: Text(
                            "Weather/Location error: $_envError",
                            style: const TextStyle(
                                fontWeight: FontWeight.w800, color: Colors.red),
                          ),
                        ),

                      // ===== ENV METRICS =====
                      Row(
                        children: [
                          Expanded(
                            child: _metricCard(
                              title: "Weather Temp",
                              value: _loadingEnv
                                  ? "…"
                                  : "${(_weather?.temperatureC ?? 0).toStringAsFixed(0)}°C",
                              icon: Icons.thermostat_rounded,
                              accent: AppColors.primary.withOpacity(0.14),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: _metricCard(
                              title: "Sensor Humidity",
                              value: soil == null
                                  ? "—"
                                  : "${soil.humidity.toStringAsFixed(0)}%",
                              icon: Icons.water_drop_rounded,
                              accent: AppColors.primary.withOpacity(0.14),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      _metricWideCard(
                        title: "Weather",
                        value: _loadingEnv
                            ? "Loading..."
                            : "${_weather?.weatherLabel ?? "—"}  • wind ${(_weather?.windSpeed ?? 0).toStringAsFixed(0)} km/h",
                        icon: Icons.wb_cloudy_rounded,
                      ),

                      const SizedBox(height: 14),

                      // ===== Soil section =====
                      Row(
                        children: [
                          Expanded(
                              child: _sectionLabel("SOIL VALUES (FIRESTORE)")),
                          if (soil != null)
                            _soilHealthChip(soil), // ✅ nice badge
                        ],
                      ),
                      const SizedBox(height: 10),

                      // Loading state
                      if (snap.connectionState == ConnectionState.waiting &&
                          soil == null)
                        _skeletonGrid()
                      else if (soil == null)
                        _emptyStateCard()
                      else
                        AnimatedOpacity(
                          duration: const Duration(milliseconds: 250),
                          opacity: 1,
                          child: GridView(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            gridDelegate:
                                const SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 2,
                              crossAxisSpacing: 12,
                              mainAxisSpacing: 12,
                              childAspectRatio: 1.25,
                            ),
                            children: [
                              _soilCard(
                                title: "Nitrogen (N)",
                                value: soil.n.toStringAsFixed(0),
                                unit: "mg/kg",
                                icon: Icons.science_rounded,
                              ),
                              _soilCard(
                                title: "Phosphorous (P)",
                                value: soil.p.toStringAsFixed(0),
                                unit: "mg/kg",
                                icon: Icons.bubble_chart_rounded,
                              ),
                              _soilCard(
                                title: "Potassium (K)",
                                value: soil.k.toStringAsFixed(0),
                                unit: "mg/kg",
                                icon: Icons.grass_rounded,
                              ),
                              _soilCard(
                                title: "Soil pH",
                                value: soil.ph.toStringAsFixed(1),
                                unit: "pH",
                                icon: Icons.show_chart_rounded,
                              ),
                              _soilCard(
                                title: "Soil Temp",
                                value: soil.temperature.toStringAsFixed(1),
                                unit: "°C",
                                icon: Icons.thermostat_rounded,
                              ),
                              _soilCard(
                                title: "Moisture",
                                value: soil.moisture.toStringAsFixed(0),
                                unit: "%",
                                icon: Icons.water_rounded,
                              ),
                            ],
                          ),
                        ),

                      if (soil?.updatedAt != null) ...[
                        const SizedBox(height: 12),
                        Text(
                          "Last update: ${soil!.updatedAt}",
                          style: TextStyle(
                            color: AppColors.textDark.withOpacity(0.55),
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ],
                    ],
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ====== UX helpers ======

  Widget _headerPill({required IconData icon, required String text}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: AppColors.primary, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withOpacity(0.90),
                fontWeight: FontWeight.w800,
                fontSize: 12.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _statusChip(String text, {required Color color}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.16),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(
            text,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w900,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _soilHealthChip(SoilReading soil) {
    // very simple “health” label (no strict science, just UX)
    String label = "OK";
    Color c = const Color(0xFF2FA36B);

    if (soil.ph < 5.5 || soil.ph > 7.8) {
      label = "pH Alert";
      c = const Color(0xFFE05C5C);
    } else if (soil.moisture < 25) {
      label = "Dry";
      c = const Color(0xFFC9A80B);
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: c.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: c.withOpacity(0.35)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: c, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(
            label,
            style: TextStyle(
              color: c,
              fontWeight: FontWeight.w900,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _skeletonGrid() {
    Widget skel() => Container(
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
          ),
        );

    return GridView(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1.25,
      ),
      children: List.generate(6, (_) => skel()),
    );
  }

  Widget _emptyStateCard() {
    return _whiteCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.red.withOpacity(0.25)),
                ),
                child: const Icon(Icons.sensors_off_rounded, color: Colors.red),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Text(
                  "No Soil Data Found",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 15,
                    color: AppColors.textDark,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            "ESP8266 must write to:\nusers/<uid>/soil_readings/latest",
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.70),
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            height: 46,
            child: OutlinedButton.icon(
              onPressed: _loadEnv,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text(
                "Retry",
                style: TextStyle(fontWeight: FontWeight.w900),
              ),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppColors.darkGreen,
                side: const BorderSide(color: AppColors.border),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                backgroundColor: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===== UI helpers (reused style) =====
  static Widget _sectionLabel(String text) {
    return Text(
      text,
      style: TextStyle(
        color: AppColors.textDark.withOpacity(0.55),
        fontWeight: FontWeight.w900,
        fontSize: 11.5,
        letterSpacing: 1.2,
      ),
    );
  }

  static Widget _whiteCard({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  static Widget _metricCard({
    required String title,
    required String value,
    required IconData icon,
    required Color accent,
  }) {
    return _whiteCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: accent,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: AppColors.darkGreen, size: 20),
          ),
          const SizedBox(height: 10),
          Text(
            title,
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.60),
              fontWeight: FontWeight.w800,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: const TextStyle(
              color: AppColors.textDark,
              fontWeight: FontWeight.w900,
              fontSize: 18,
            ),
          ),
        ],
      ),
    );
  }

  static Widget _metricWideCard({
    required String title,
    required String value,
    required IconData icon,
  }) {
    return _whiteCard(
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.16),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: AppColors.darkGreen, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: AppColors.textDark.withOpacity(0.60),
                    fontWeight: FontWeight.w800,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  value,
                  style: const TextStyle(
                    color: AppColors.textDark,
                    fontWeight: FontWeight.w900,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  static Widget _soilCard({
    required String title,
    required String value,
    required String unit,
    required IconData icon,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.10),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.border),
            ),
            child: Icon(icon, color: AppColors.darkGreen, size: 20),
          ),
          const SizedBox(height: 10),
          Text(
            title,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.70),
              fontWeight: FontWeight.w800,
              fontSize: 12,
            ),
          ),
          const Spacer(),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                value,
                style: const TextStyle(
                  color: AppColors.darkGreen,
                  fontWeight: FontWeight.w900,
                  fontSize: 20,
                ),
              ),
              const SizedBox(width: 6),
              Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: Text(
                  unit,
                  style: TextStyle(
                    color: AppColors.textDark.withOpacity(0.55),
                    fontWeight: FontWeight.w800,
                    fontSize: 11.5,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
