import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/api_client.dart';

class LaborApi {
  static const Duration _timeout = Duration(seconds: 20);

  // ✅ ngrok + JSON friendly headers
  static const Map<String, String> _getHeaders = {
    "Accept": "application/json",
    "ngrok-skip-browser-warning": "true",
  };

  static const Map<String, String> _postHeaders = {
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
        "Check: ngrok HTTPS URL / route path / backend running.\n"
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
  }) async {
    final uri = ApiClient.url(path);

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
          "Check ngrok URL, HTTPS, and backend route.\n"
          "Body: ${_shortBody(res.body)}",
        );
      }
      throw Exception(
          "$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}");
    }

    return res;
  }

  static Future<http.Response> _post(
    String path, {
    required String endpointName,
    required Map<String, dynamic> body,
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

      // try to parse JSON error safely
      try {
        final errMap = jsonDecode(res.body) as Map<String, dynamic>;
        throw Exception(errMap["error"]?.toString() ?? "$endpointName failed");
      } catch (_) {
        throw Exception(
            "$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}");
      }
    }

    return res;
  }

  // ✅ GET /api/labour/locations
  static Future<List<String>> getLocations() async {
    final res = await _get(
      "/api/labour/locations",
      endpointName: "Labour locations",
    );

    final map = _decodeJsonObject(res, endpointName: "Labour locations");
    final list =
        (map["locations"] as List? ?? []).map((e) => e.toString()).toList();

    // unique + sorted + non-empty
    final unique = list.where((e) => e.trim().isNotEmpty).toSet().toList()
      ..sort();
    return unique;
  }

  // ✅ GET /api/labour/skills
  static Future<List<String>> getSkills() async {
    final res = await _get(
      "/api/labour/skills",
      endpointName: "Labour skills",
    );

    final map = _decodeJsonObject(res, endpointName: "Labour skills");
    final list =
        (map["skills"] as List? ?? []).map((e) => e.toString()).toList();

    final unique = list.where((e) => e.trim().isNotEmpty).toSet().toList()
      ..sort();
    return unique;
  }

  // ✅ POST /api/labour/search
  static Future<Map<String, dynamic>> search({
    required String query,
    required String location,
    String? skill,
    int topK = 30,
  }) async {
    final res = await _post(
      "/api/labour/search",
      endpointName: "Labour search",
      body: {
        "query": query,
        "location": location,
        "skill": (skill == null || skill.trim().isEmpty) ? null : skill,
        "top_k": topK,
      },
    );

    return _decodeJsonObject(res, endpointName: "Labour search");
  }
}

/// Route arguments for labour search / list screens.
class LaborSearchArgs {
  final String query;
  final String location;
  final String? skill;

  LaborSearchArgs({
    required this.query,
    required this.location,
    this.skill,
  });
}
