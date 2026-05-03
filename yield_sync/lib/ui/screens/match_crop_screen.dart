import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';

import '../../services/crop_predict_service.dart';
import '../../services/prediction_history_service.dart';

// ✅ NEW: fetch recommendation + open screen
import '../../services/crop_recommendation_service.dart';
import '../screens/crop_recommendation_screen.dart';

class MatchCropScreen extends StatefulWidget {
  const MatchCropScreen({super.key});

  @override
  State<MatchCropScreen> createState() => _MatchCropScreenState();
}

class _MatchCropScreenState extends State<MatchCropScreen> {
  final _nCtrl = TextEditingController();
  final _pCtrl = TextEditingController();
  final _kCtrl = TextEditingController();
  final _phCtrl = TextEditingController();
  final _tempCtrl = TextEditingController();
  final _humCtrl = TextEditingController();

  String _place = "Tap “Use Live Location”";
  double? _lat;
  double? _lon;

  bool _loadingLoc = false;
  bool _loadingPredict = false;

  CropPredictResult? _result;

  @override
  void dispose() {
    _nCtrl.dispose();
    _pCtrl.dispose();
    _kCtrl.dispose();
    _phCtrl.dispose();
    _tempCtrl.dispose();
    _humCtrl.dispose();
    super.dispose();
  }

  // ✅ ONLY location (lat/lon + place)
  Future<void> _useLiveLocation() async {
    setState(() {
      _loadingLoc = true;
      _result = null;
    });

    try {
      final enabled = await Geolocator.isLocationServiceEnabled();
      if (!enabled) throw Exception("Location service is disabled");

      var perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.deniedForever) {
        throw Exception("Location permission denied forever");
      }
      if (perm == LocationPermission.denied) {
        throw Exception("Location permission denied");
      }

      final pos = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      _lat = pos.latitude;
      _lon = pos.longitude;

      // city name
      final placemarks = await placemarkFromCoordinates(_lat!, _lon!);
      final pm = placemarks.isNotEmpty ? placemarks.first : null;
      final city = [
        pm?.locality,
        pm?.administrativeArea,
      ].where((e) => (e ?? "").trim().isNotEmpty).join(", ");

      if (!mounted) return;
      setState(() => _place = city.isEmpty ? "Current location" : city);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Location error: $e")),
      );
    } finally {
      if (mounted) setState(() => _loadingLoc = false);
    }
  }

  Future<void> _analyze() async {
    final n = double.tryParse(_nCtrl.text.trim());
    final p = double.tryParse(_pCtrl.text.trim());
    final k = double.tryParse(_kCtrl.text.trim());
    final ph = double.tryParse(_phCtrl.text.trim());
    final temp = double.tryParse(_tempCtrl.text.trim());
    final humidity = double.tryParse(_humCtrl.text.trim());

    if (_lat == null || _lon == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Use Live Location first.")),
      );
      return;
    }

    if (n == null ||
        p == null ||
        k == null ||
        ph == null ||
        temp == null ||
        humidity == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Enter valid N, P, K, pH, Temperature, Humidity."),
        ),
      );
      return;
    }

    setState(() {
      _loadingPredict = true;
      _result = null;
    });

    try {
      // ✅ 1) Predict
      final res = await CropPredictService.predict(
        n: n,
        p: p,
        k: k,
        ph: ph,
        temp: temp,
        humidity: humidity,
        lat: _lat!,
        lon: _lon!,
      );

      if (!mounted) return;
      setState(() => _result = res);

      // ✅ 2) SAVE to Firestore History (your existing logic)
      final request = {
        "N": n,
        "P": p,
        "K": k,
        "ph": ph,
        "temperature": temp,
        "humidity": humidity,
        "lat": _lat,
        "lon": _lon,
      };

      await PredictionHistoryService().saveCropPrediction(
        request: request,
        response: res.toJson(),
        lat: _lat!,
        lon: _lon!,
        place: _place,
      );

      // ✅ 3) Fetch crop recommendation from Firestore (NEW)
      final rec =
          await CropRecommendationService.fetchByPrediction(res.prediction);

      if (!mounted) return;

      if (rec == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text(
                  "Saved ✅ but no recommendation found for: ${res.prediction}")),
        );
        return;
      }

      // ✅ 4) Open recommendation screen (NEW)
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => CropRecommendationScreen(data: rec),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Prediction/Save error: $e")),
      );
    } finally {
      if (mounted) setState(() => _loadingPredict = false);
    }
  }

  void _clearAll() {
    _nCtrl.clear();
    _pCtrl.clear();
    _kCtrl.clear();
    _phCtrl.clear();
    _tempCtrl.clear();
    _humCtrl.clear();
    setState(() {
      _result = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final hasLocation = _lat != null && _lon != null;

    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===== Modern Header =====
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
                  top: 26,
                  child: Opacity(
                    opacity: 0.10,
                    child:
                        Icon(Icons.spa_rounded, size: 150, color: Colors.white),
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.arrow_back_ios_new_rounded),
                          color: Colors.white,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          "Match Crop",
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.92),
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                        const Spacer(),

                        // ✅ profile click
                        InkWell(
                          borderRadius: BorderRadius.circular(999),
                          onTap: () =>
                              Navigator.pushNamed(context, AppRoutes.profile),
                          child: CircleAvatar(
                            radius: 18,
                            backgroundColor: Colors.white.withOpacity(0.12),
                            child: const Icon(Icons.person_rounded,
                                color: Colors.white),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Find best crop for your soil",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.96),
                        fontWeight: FontWeight.w900,
                        fontSize: 22,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      "Use live location + enter NPK, pH, temp & humidity",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontWeight: FontWeight.w600,
                        fontSize: 12.8,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // ✅ location chips row
                    Row(
                      children: [
                        Expanded(
                          child: _miniChip(
                            icon: Icons.location_city_rounded,
                            label: _place,
                            full: true,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        Expanded(
                          child: _miniChip(
                            icon: Icons.location_on_rounded,
                            label: !hasLocation
                                ? "Lat/Lon —"
                                : "Lat ${_lat!.toStringAsFixed(5)} • Lon ${_lon!.toStringAsFixed(5)}",
                            full: true,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          Expanded(
            child: RefreshIndicator(
              onRefresh: () async {
                await _useLiveLocation();
              },
              child: ListView(
                physics: const BouncingScrollPhysics(),
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                children: [
                  // ===== Modern action row =====
                  Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 48,
                          child: ElevatedButton.icon(
                            onPressed: _loadingLoc ? null : _useLiveLocation,
                            icon: const Icon(Icons.my_location_rounded),
                            label: Text(
                              _loadingLoc ? "Loading..." : "Use Live Location",
                              style:
                                  const TextStyle(fontWeight: FontWeight.w900),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: AppColors.darkGreen,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      SizedBox(
                        height: 48,
                        width: 56,
                        child: OutlinedButton(
                          onPressed: _clearAll,
                          style: OutlinedButton.styleFrom(
                            foregroundColor: AppColors.textDark,
                            side: const BorderSide(color: AppColors.border),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            backgroundColor: Colors.white,
                          ),
                          child: const Icon(Icons.delete_outline_rounded),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 12),

                  // ===== Inputs =====
                  Container(
                    padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(22),
                      border: Border.all(color: AppColors.border),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.06),
                          blurRadius: 22,
                          offset: const Offset(0, 14),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 36,
                              height: 36,
                              decoration: BoxDecoration(
                                color: AppColors.primary.withOpacity(0.16),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: const Icon(Icons.tune_rounded,
                                  color: AppColors.darkGreen),
                            ),
                            const SizedBox(width: 10),
                            const Expanded(
                              child: Text(
                                "Enter Values",
                                style: TextStyle(
                                  fontWeight: FontWeight.w900,
                                  fontSize: 16,
                                  color: AppColors.textDark,
                                ),
                              ),
                            ),
                            if (hasLocation)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 6),
                                decoration: BoxDecoration(
                                  color: AppColors.primary.withOpacity(0.10),
                                  borderRadius: BorderRadius.circular(999),
                                  border: Border.all(color: AppColors.border),
                                ),
                                child: const Text(
                                  "Location OK",
                                  style: TextStyle(
                                    fontWeight: FontWeight.w900,
                                    color: AppColors.darkGreen,
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Text(
                          "Tip: If you don’t know values, you can use soil test results.",
                          style: TextStyle(
                            color: AppColors.textDark.withOpacity(0.55),
                            fontWeight: FontWeight.w700,
                            fontSize: 12.2,
                          ),
                        ),
                        const SizedBox(height: 14),
                        Row(
                          children: [
                            Expanded(child: _numField("N", _nCtrl)),
                            const SizedBox(width: 10),
                            Expanded(child: _numField("P", _pCtrl)),
                            const SizedBox(width: 10),
                            Expanded(child: _numField("K", _kCtrl)),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(child: _numField("pH", _phCtrl)),
                            const SizedBox(width: 10),
                            Expanded(child: _numField("Temp (°C)", _tempCtrl)),
                          ],
                        ),
                        const SizedBox(height: 12),
                        _numField("Humidity (%)", _humCtrl),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          height: 52,
                          child: ElevatedButton.icon(
                            onPressed: (!hasLocation || _loadingPredict)
                                ? null
                                : _analyze,
                            icon: const Icon(Icons.auto_graph_rounded),
                            label: Text(
                              _loadingPredict
                                  ? "Analyzing..."
                                  : "Analyze & Recommend",
                              style:
                                  const TextStyle(fontWeight: FontWeight.w900),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: AppColors.darkGreen,
                              disabledBackgroundColor:
                                  AppColors.border.withOpacity(0.6),
                              disabledForegroundColor:
                                  AppColors.textDark.withOpacity(0.55),
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 12),

                  // Optional: you can keep showing result card
                  if (_result != null) _resultCard(_result!),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ✅ ONLY show crop name (no probability list)
  Widget _resultCard(CropPredictResult r) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.16),
                  borderRadius: BorderRadius.circular(12),
                ),
                child:
                    const Icon(Icons.spa_rounded, color: AppColors.darkGreen),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text(
                  "Recommended Crop",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 16,
                    color: AppColors.textDark,
                  ),
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: AppColors.border),
                ),
                child: const Text(
                  "Result",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.darkGreen,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.14),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(
              children: [
                const Icon(Icons.verified_rounded,
                    color: AppColors.darkGreen, size: 26),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    r.prediction,
                    style: const TextStyle(
                      fontWeight: FontWeight.w900,
                      fontSize: 20,
                      color: AppColors.textDark,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(
              children: [
                Icon(Icons.info_outline_rounded,
                    size: 18, color: AppColors.textDark.withOpacity(0.70)),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    "Saved to your History. Recommendation screen opened.",
                    style: TextStyle(
                      color: AppColors.textDark.withOpacity(0.70),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _numField(String label, TextEditingController ctrl) {
    return TextField(
      controller: ctrl,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(
        labelText: label,
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.primary, width: 1.6),
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      ),
    );
  }

  Widget _miniChip(
      {required IconData icon, required String label, bool full = false}) {
    return Container(
      width: full ? double.infinity : null,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Row(
        mainAxisSize: full ? MainAxisSize.max : MainAxisSize.min,
        children: [
          Icon(icon, color: Colors.white.withOpacity(0.95), size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
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
}
