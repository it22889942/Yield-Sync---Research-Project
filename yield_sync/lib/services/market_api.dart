import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/api_client.dart';

class MarketApi {
  static const Duration _timeout = Duration(seconds: 25);
  static const Duration _updateTimeout = Duration(minutes: 5);
  static const Duration _retrainTimeout = Duration(minutes: 40);

  // ✅ ngrok + JSON-safe headers
  static const Map<String, String> _getHeaders = {
    "Accept": "application/json",
    "ngrok-skip-browser-warning": "true",
  };

  static const Map<String, String> _postHeaders = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "ngrok-skip-browser-warning": "true",
  };

  static String _shortBody(String body, {int max = 260}) {
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

  static Map<String, dynamic> _decodeJsonObject(
    http.Response res, {
    required String endpointName,
  }) {
    final body = res.body;

    if (_looksLikeHtml(body)) {
      throw Exception(
        "$endpointName returned HTML instead of JSON.\n"
        "Status: ${res.statusCode}\n"
        "Content-Type: ${res.headers['content-type']}\n"
        "Check: ngrok URL (HTTPS), backend route, and backend server.\n"
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

  static Future<http.Response> _get(
    String path, {
    required String endpointName,
    Map<String, String>? queryParameters,
  }) async {
    final uri = ApiClient.url(path).replace(queryParameters: queryParameters);

    http.Response res;
    try {
      res = await http.get(uri, headers: _getHeaders).timeout(_timeout);
    } on TimeoutException {
      throw Exception("$endpointName request timeout");
    } catch (e) {
      throw Exception("$endpointName request failed: $e");
    }

    // ignore: avoid_print
    print("GET $uri -> ${res.statusCode} | ${res.headers['content-type']}");

    if (res.statusCode != 200) {
      if (_looksLikeHtml(res.body)) {
        throw Exception(
          "$endpointName failed (${res.statusCode}) and returned HTML, not JSON.\n"
          "Check ngrok HTTPS URL, API path, and backend logs.\n"
          "Body: ${_shortBody(res.body)}",
        );
      }

      // try JSON error parse
      try {
        final errMap = jsonDecode(res.body) as Map<String, dynamic>;
        throw Exception(errMap["error"]?.toString() ?? "$endpointName failed");
      } catch (_) {
        throw Exception(
          "$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}",
        );
      }
    }

    return res;
  }

  static Future<http.Response> _post(
    String path, {
    required String endpointName,
    required Map<String, dynamic> body,
    Duration? timeout,
  }) async {
    final uri = ApiClient.url(path);

    http.Response res;
    try {
      res = await http
          .post(
            uri,
            headers: _postHeaders,
            body: jsonEncode(body),
          )
          .timeout(timeout ?? _timeout);
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
          "Check ngrok HTTPS URL, route path, and backend logs.\n"
          "Body: ${_shortBody(res.body)}",
        );
      }

      try {
        final errMap = jsonDecode(res.body) as Map<String, dynamic>;
        throw Exception(errMap["error"]?.toString() ?? "$endpointName failed");
      } catch (_) {
        throw Exception(
          "$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}",
        );
      }
    }

    return res;
  }

  // GET /api/yieldsync/crops
  static Future<Map<String, dynamic>> getCrops() async {
    final res = await _get(
      "/api/yieldsync/crops",
      endpointName: "Get crops",
    );
    return _decodeJsonObject(res, endpointName: "Get crops");
  }

  // GET /api/yieldsync/markets?crop=Rice
  static Future<List<String>> getMarkets(String crop) async {
    final res = await _get(
      "/api/yieldsync/markets",
      endpointName: "Get markets",
      queryParameters: {"crop": crop},
    );

    final map = _decodeJsonObject(res, endpointName: "Get markets");
    final list =
        (map["markets"] as List? ?? []).map((e) => e.toString()).toList();

    final unique = list.where((e) => e.trim().isNotEmpty).toSet().toList()
      ..sort();
    return unique;
  }

  // POST /api/yieldsync/predict
  static Future<Map<String, dynamic>> predict({
    required String crop,
    required String market,
    required int daysAhead,
  }) async {
    final res = await _post(
      "/api/yieldsync/predict",
      endpointName: "Predict price",
      body: {
        "crop": crop,
        "market": market,
        "days_ahead": daysAhead,
      },
    );

    return _decodeJsonObject(res, endpointName: "Predict price");
  }

  // POST /api/yieldsync/recommendation
  static Future<Map<String, dynamic>> recommendation({
    required String crop,
    required String market,
    required int daysAhead,
    required double quantityKg,
    int daysSinceHarvest = 0,
    double? transportCostPerKg,
    double? storageCostPerKgDay,
    double? fixedCostTotal,
    double? spoilageRate,
  }) async {
    final body = <String, dynamic>{
      "crop": crop,
      "market": market,
      "days_ahead": daysAhead,
      "quantity_kg": quantityKg,
      "days_since_harvest": daysSinceHarvest,
    };
    if (transportCostPerKg != null) body["transport_cost_per_kg"] = transportCostPerKg;
    if (storageCostPerKgDay != null) body["storage_cost_per_kg_day"] = storageCostPerKgDay;
    if (fixedCostTotal != null) body["fixed_cost_total"] = fixedCostTotal;
    if (spoilageRate != null) body["spoilage_rate"] = spoilageRate;

    final res = await _post(
      "/api/yieldsync/recommendation",
      endpointName: "Get recommendation",
      body: body,
    );

    return _decodeJsonObject(res, endpointName: "Get recommendation");
  }

  // GET /api/yieldsync/history?crop=Rice&market=Colombo&days=30
  static Future<Map<String, dynamic>> history({
    required String crop,
    required String market,
    int days = 30,
  }) async {
    final res = await _get(
      "/api/yieldsync/history",
      endpointName: "Get history",
      queryParameters: {
        "crop": crop,
        "market": market,
        "days": "$days",
      },
    );

    return _decodeJsonObject(res, endpointName: "Get history");
  }

  // GET /api/yieldsync/avg-by-market?crop=Rice&days=7
  static Future<Map<String, dynamic>> avgByMarket({
    required String crop,
    int days = 7,
  }) async {
    final res = await _get(
      "/api/yieldsync/avg-by-market",
      endpointName: "Average by market",
      queryParameters: {
        "crop": crop,
        "days": "$days",
      },
    );

    return _decodeJsonObject(res, endpointName: "Average by market");
  }

  // GET /api/yieldsync/trends?crop=All&market=All&max_points=400&recent=50
  static Future<Map<String, dynamic>> trends({
    String crop = "All",
    String market = "All",
    int maxPoints = 400,
    int recent = 50,
  }) async {
    final res = await _get(
      "/api/yieldsync/trends",
      endpointName: "Get trends",
      queryParameters: {
        "crop": crop,
        "market": market,
        "max_points": "$maxPoints",
        "recent": "$recent",
      },
    );

    return _decodeJsonObject(res, endpointName: "Get trends");
  }

  // ✅ Forecast & Analytics chart data (multi-line)
  // GET /api/yieldsync/trends?crop=All&market=All&max_points=420&recent=0
  static Future<Map<String, dynamic>> forecastAnalytics({
    String crop = "All",
    String market = "All",
    int maxPoints = 420,
  }) async {
    final res = await _get(
      "/api/yieldsync/trends",
      endpointName: "Forecast analytics",
      queryParameters: {
        "crop": crop,
        "market": market,
        "max_points": "$maxPoints",
        "recent": "0",
      },
    );

    return _decodeJsonObject(res, endpointName: "Forecast analytics");
  }

  // POST /api/yieldsync/data/update-smart
  static Future<Map<String, dynamic>> updateSmartWeek({
    String? startDate,
    String? dataPath,
  }) async {
    final body = <String, dynamic>{};
    if (startDate != null && startDate.trim().isNotEmpty) {
      body["start_date"] = startDate.trim();
    }
    if (dataPath != null && dataPath.trim().isNotEmpty) {
      body["data_path"] = dataPath.trim();
    }

    final res = await _post(
      "/api/yieldsync/data/update-smart",
      endpointName: "Update latest week",
      body: body,
      timeout: _updateTimeout,
    );

    return _decodeJsonObject(res, endpointName: "Update latest week");
  }

  // POST /api/yieldsync/models/retrain
  static Future<Map<String, dynamic>> retrainModels({
    String? dataPath,
    String? saveDir,
  }) async {
    final body = <String, dynamic>{};
    if (dataPath != null && dataPath.trim().isNotEmpty) {
      body["data_path"] = dataPath.trim();
    }
    if (saveDir != null && saveDir.trim().isNotEmpty) {
      body["save_dir"] = saveDir.trim();
    }

    final res = await _post(
      "/api/yieldsync/models/retrain",
      endpointName: "Retrain models",
      body: body,
      timeout: _retrainTimeout,
    );

    return _decodeJsonObject(res, endpointName: "Retrain models");
  }
}
