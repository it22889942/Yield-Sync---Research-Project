import 'dart:convert';
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;

class LocationResult {
  final double lat;
  final double lon;
  final String label;

  LocationResult({required this.lat, required this.lon, required this.label});
}

class LocationService {
  static Future<LocationResult> getCurrentLocationLabel() async {
    // 1) Check service
    final enabled = await Geolocator.isLocationServiceEnabled();
    if (!enabled) throw Exception("Location service OFF");

    // 2) Permission
    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) {
      perm = await Geolocator.requestPermission();
    }
    if (perm == LocationPermission.denied ||
        perm == LocationPermission.deniedForever) {
      throw Exception("Location permission denied");
    }

    // 3) Position
    final pos = await Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.high,
    );

    // Default fallback: coordinates
    String label =
        "${pos.latitude.toStringAsFixed(4)}, ${pos.longitude.toStringAsFixed(4)}";

    // 4) Try device geocoding (fast)
    try {
      final places =
          await placemarkFromCoordinates(pos.latitude, pos.longitude);
      if (places.isNotEmpty) {
        final p = places.first;

        final parts = <String>[
          if ((p.subLocality ?? "").trim().isNotEmpty) p.subLocality!.trim(),
          if ((p.locality ?? "").trim().isNotEmpty) p.locality!.trim(),
          if ((p.administrativeArea ?? "").trim().isNotEmpty)
            p.administrativeArea!.trim(),
          if ((p.country ?? "").trim().isNotEmpty) p.country!.trim(),
        ];

        final joined = parts.where((e) => e.isNotEmpty).toList().join(", ");
        if (joined.isNotEmpty) label = joined;
      }
    } catch (_) {
      // ignore and fallback below
    }

    // 5) If still coords (no place name), use Nominatim (free, no key)
    final isStillCoords = label.contains(",") && label.contains(".");
    if (isStillCoords) {
      final nom = await _reverseGeocodeNominatim(pos.latitude, pos.longitude);
      if (nom.isNotEmpty) label = nom;
    }

    return LocationResult(lat: pos.latitude, lon: pos.longitude, label: label);
  }

  static Future<String> _reverseGeocodeNominatim(double lat, double lon) async {
    try {
      final uri = Uri.parse(
        "https://nominatim.openstreetmap.org/reverse"
        "?format=jsonv2&lat=$lat&lon=$lon&zoom=14&addressdetails=1",
      );

      // IMPORTANT: Nominatim requires a User-Agent
      final res = await http.get(uri, headers: {
        "User-Agent": "YieldSyncApp/1.0 (contact: test1@gmail.com)",
        "Accept": "application/json",
      });

      if (res.statusCode != 200) return "";

      final data = jsonDecode(res.body) as Map<String, dynamic>;
      final display = (data["display_name"] ?? "").toString().trim();
      if (display.isEmpty) return "";

      // Make it shorter (optional)
      final parts = display.split(",").map((e) => e.trim()).toList();
      if (parts.length >= 3) {
        return "${parts[0]}, ${parts[1]}, ${parts.last}"; // ex: "Kegalle, Sabaragamuwa, Sri Lanka"
      }
      return display;
    } catch (_) {
      return "";
    }
  }
}
