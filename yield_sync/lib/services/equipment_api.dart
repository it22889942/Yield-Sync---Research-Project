import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

import '../config/api_client.dart';

class EquipmentApi {
  static const Duration _timeout = Duration(seconds: 20);

  // Common headers for ngrok + JSON APIs
  static Map<String, String> get _jsonHeaders => const {
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true",
      };

  static Map<String, String> get _jsonPostHeaders => const {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true",
      };

  static String _shortBody(String body, {int max = 240}) {
    final clean = body.replaceAll(RegExp(r"\s+"), " ").trim();
    if (clean.length <= max) return clean;
    return "${clean.substring(0, max)}...";
  }

  static bool _looksLikeHtml(String body) {
    final b = body.trimLeft().toLowerCase();
    return b.startsWith("<!doctype html") ||
        b.startsWith("<html") ||
        b.startsWith("<!doctype");
  }

  static Map<String, dynamic> _decodeJsonObject(http.Response res,
      {required String endpointName}) {
    final body = res.body;

    // Helpful detection for ngrok / proxy / 404 HTML pages
    if (_looksLikeHtml(body)) {
      throw Exception(
        "$endpointName returned HTML instead of JSON.\n"
        "Status: ${res.statusCode}\n"
        "Content-Type: ${res.headers['content-type']}\n"
        "This usually means: wrong URL / ngrok HTTP instead of HTTPS / ngrok warning page / wrong route.\n"
        "Body: ${_shortBody(body)}",
      );
    }

    dynamic decoded;
    try {
      decoded = jsonDecode(body);
    } catch (e) {
      throw Exception(
        "$endpointName JSON parse failed: $e\n"
        "Status: ${res.statusCode}\n"
        "Content-Type: ${res.headers['content-type']}\n"
        "Body: ${_shortBody(body)}",
      );
    }

    if (decoded is! Map<String, dynamic>) {
      throw Exception(
        "$endpointName expected JSON object but got ${decoded.runtimeType}. "
        "Body: ${_shortBody(body)}",
      );
    }

    return decoded;
  }

  static Future<http.Response> _get(Uri uri,
      {required String endpointName}) async {
    http.Response res;
    try {
      res = await http.get(uri, headers: _jsonHeaders).timeout(_timeout);
    } on TimeoutException {
      throw Exception("$endpointName request timeout");
    } catch (e) {
      throw Exception("$endpointName request failed: $e");
    }

    // Debug logs (very useful while fixing ngrok issues)
    // ignore: avoid_print
    print("GET $uri -> ${res.statusCode} | ${res.headers['content-type']}");

    if (res.statusCode != 200) {
      if (_looksLikeHtml(res.body)) {
        throw Exception(
          "$endpointName failed (${res.statusCode}) and returned HTML, not JSON.\n"
          "Check ngrok URL, HTTPS, and backend route.\n"
          "Body: ${_shortBody(res.body)}",
        );
      }
      throw Exception(
          "$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}");
    }

    return res;
  }

  static Future<http.Response> _post(Uri uri,
      {required String endpointName,
      required Map<String, dynamic> body}) async {
    http.Response res;
    try {
      res = await http
          .post(
            uri,
            headers: _jsonPostHeaders,
            body: jsonEncode(body),
          )
          .timeout(_timeout);
    } on TimeoutException {
      throw Exception("$endpointName request timeout");
    } catch (e) {
      throw Exception("$endpointName request failed: $e");
    }

    // ignore: avoid_print
    print("POST $uri -> ${res.statusCode} | ${res.headers['content-type']}");

    if (res.statusCode != 200) {
      if (_looksLikeHtml(res.body)) {
        throw Exception(
          "$endpointName failed (${res.statusCode}) and returned HTML, not JSON.\n"
          "Check ngrok URL, HTTPS, route path, and backend logs.\n"
          "Body: ${_shortBody(res.body)}",
        );
      }
      throw Exception(
          "$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}");
    }

    return res;
  }

  // ✅ GET /api/equipment/locations
  static Future<List<String>> getLocations() async {
    final uri = ApiClient.url("/api/equipment/locations");
    final res = await _get(uri, endpointName: "Equipment locations");

    final data = _decodeJsonObject(res, endpointName: "Equipment locations");
    final list = (data["locations"] as List<dynamic>? ?? []);
    final locs = list
        .map((e) => e.toString())
        .where((e) => e.trim().isNotEmpty)
        .toList();

    final unique = locs.toSet().toList()..sort();
    return unique;
  }

  // ✅ GET /api/equipment/types
  static Future<List<String>> getTypes() async {
    final uri = ApiClient.url("/api/equipment/types");
    final res = await _get(uri, endpointName: "Equipment types");

    final data = _decodeJsonObject(res, endpointName: "Equipment types");
    final list = (data["types"] as List<dynamic>? ?? []);
    final types = list
        .map((e) => e.toString())
        .where((e) => e.trim().isNotEmpty)
        .toList();

    final unique = types.toSet().toList()..sort();
    return unique;
  }

  // ✅ POST /api/equipment/search
  static Future<List<EquipmentListItem>> search({
    required String query,
    required String location, // main district
    String type = "", // base type for backend (Tractor)
    int topK = 30,
  }) async {
    final uri = ApiClient.url("/api/equipment/search");

    final res = await _post(
      uri,
      endpointName: "Equipment search",
      body: {
        "query": query,
        "location": location,
        "type": type,
        "top_k": topK,
      },
    );

    final data = _decodeJsonObject(res, endpointName: "Equipment search");
    final items = (data["items"] as List<dynamic>? ?? []);
    return items
        .whereType<Map>()
        .map((e) => EquipmentListItem.fromJson(Map<String, dynamic>.from(e)))
        .toList();
  }

  // ✅ GET /api/equipment/item/<id>
  static Future<EquipmentDetails> getItem(String id) async {
    final uri = ApiClient.url("/api/equipment/item/$id");
    final res = await _get(uri, endpointName: "Equipment item");

    final data = _decodeJsonObject(res, endpointName: "Equipment item");
    return EquipmentDetails.fromJson(data);
  }
}

// ===================== LIST ITEM MODEL =====================
class EquipmentListItem {
  final String id;
  final String equipmentType;
  final String forCrop;
  final String location; // main district
  final String nearestMajorDistrict;
  final double hourlyRate;
  final double dailyRate;
  final double rating;
  final int pastBookings;
  final String ownerName;
  final String ownerContact;
  final String availableDay;
  final String availableTime;
  final String condition;

  EquipmentListItem({
    required this.id,
    required this.equipmentType,
    required this.forCrop,
    required this.location,
    required this.nearestMajorDistrict,
    required this.hourlyRate,
    required this.dailyRate,
    required this.rating,
    required this.pastBookings,
    required this.ownerName,
    required this.ownerContact,
    required this.availableDay,
    required this.availableTime,
    required this.condition,
  });

  factory EquipmentListItem.fromJson(Map<String, dynamic> j) {
    double toD(v) => (v is num) ? v.toDouble() : double.tryParse('$v') ?? 0.0;
    int toI(v) => (v is num) ? v.toInt() : int.tryParse('$v') ?? 0;

    return EquipmentListItem(
      // Support both /api/equipment/search (lowercase keys)
      // and /api/recommend/recommend (DataFrame column-style keys).
      id: (j['id'] ?? j['Equipment_ID'] ?? '').toString(),
      equipmentType:
          (j['equipment_type'] ?? j['Equipment_Type'] ?? '').toString(),
      forCrop: (j['for_crop'] ?? j['For_Crop'] ?? '').toString(),
      location: (j['location'] ??
              j['Main_District'] ??
              j['Nearest_Major_District'] ??
              '')
          .toString(),
      nearestMajorDistrict:
          (j['nearest_major_district'] ?? j['Nearest_Major_District'] ?? '')
              .toString(),
      hourlyRate: toD(j['hourly_rate'] ?? j['Hourly_Rate_LKR']),
      dailyRate: toD(j['daily_rate'] ?? j['Daily_Rate_LKR']),
      rating: toD(j['rating'] ?? j['Rating']),
      pastBookings: toI(j['past_bookings'] ?? j['Past_Bookings']),
      ownerName:
          (j['owner_name'] ?? j['Equipment_Owner_Name'] ?? '—').toString(),
      ownerContact:
          (j['owner_contact'] ?? j['Owner_Contact_No'] ?? '—').toString(),
      availableDay:
          (j['available_day'] ?? j['Available_Day'] ?? '').toString(),
      availableTime:
          (j['available_time'] ?? j['Available_Time'] ?? '').toString(),
      condition: (j['condition'] ?? j['Condition'] ?? '').toString(),
    );
  }
}

// ===================== DETAILS MODEL =====================
class EquipmentDetails {
  final String id;
  final String equipmentType;
  final String forCrop;
  final double hourlyRate;
  final double dailyRate;
  final double rating;
  final int pastBookings;
  final double successRatePct;
  final double distanceKm;
  final String nearestMajorDistrict;
  final String mainDistrict;
  final String season;
  final String ownerName;
  final String ownerContact;
  final String condition;
  final int manufactureYear;
  final double powerHp;
  final double maintenanceCostMonth;
  final double avgUsageHoursMonth;
  final double downtimeDaysMonth;
  final String ownerType;
  final int ownerExperienceYears;
  final String lastBookedBy;
  final String insuranceValidTill;
  final String registrationNo;
  final String availableDay;
  final String availableTime;

  EquipmentDetails({
    required this.id,
    required this.equipmentType,
    required this.forCrop,
    required this.hourlyRate,
    required this.dailyRate,
    required this.rating,
    required this.pastBookings,
    required this.successRatePct,
    required this.distanceKm,
    required this.nearestMajorDistrict,
    required this.mainDistrict,
    required this.season,
    required this.ownerName,
    required this.ownerContact,
    required this.condition,
    required this.manufactureYear,
    required this.powerHp,
    required this.maintenanceCostMonth,
    required this.avgUsageHoursMonth,
    required this.downtimeDaysMonth,
    required this.ownerType,
    required this.ownerExperienceYears,
    required this.lastBookedBy,
    required this.insuranceValidTill,
    required this.registrationNo,
    required this.availableDay,
    required this.availableTime,
  });

  factory EquipmentDetails.fromJson(Map<String, dynamic> j) {
    double toD(v) => (v is num) ? v.toDouble() : double.tryParse("$v") ?? 0.0;
    int toI(v) => (v is num) ? v.toInt() : int.tryParse("$v") ?? 0;

    return EquipmentDetails(
      id: (j["id"] ?? "").toString(),
      equipmentType: (j["equipment_type"] ?? "").toString(),
      forCrop: (j["for_crop"] ?? "").toString(),
      hourlyRate: toD(j["hourly_rate"]),
      dailyRate: toD(j["daily_rate"]),
      rating: toD(j["rating"]),
      pastBookings: toI(j["past_bookings"]),
      successRatePct: toD(j["success_rate_pct"]),
      distanceKm: toD(j["distance_km"]),
      nearestMajorDistrict: (j["nearest_major_district"] ?? "").toString(),
      mainDistrict: (j["main_district"] ?? "").toString(),
      season: (j["season"] ?? "").toString(),
      ownerName: (j["owner_name"] ?? "—").toString(),
      ownerContact: (j["owner_contact"] ?? "—").toString(),
      condition: (j["condition"] ?? "").toString(),
      manufactureYear: toI(j["manufacture_year"]),
      powerHp: toD(j["power_hp"]),
      maintenanceCostMonth: toD(j["maintenance_cost_month"]),
      avgUsageHoursMonth: toD(j["avg_usage_hours_month"]),
      downtimeDaysMonth: toD(j["downtime_days_month"]),
      ownerType: (j["owner_type"] ?? "").toString(),
      ownerExperienceYears: toI(j["owner_experience_years"]),
      lastBookedBy: (j["last_booked_by"] ?? "").toString(),
      insuranceValidTill: (j["insurance_valid_till"] ?? "").toString(),
      registrationNo: (j["registration_no"] ?? "").toString(),
      availableDay: (j["available_day"] ?? "").toString(),
      availableTime: (j["available_time"] ?? "").toString(),
    );
  }
}
