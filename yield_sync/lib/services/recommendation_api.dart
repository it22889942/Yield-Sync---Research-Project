import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_client.dart';

class RecommendationApi {
  RecommendationApi._();

  static const Duration _timeout = Duration(seconds: 25);

  static const Map<String, String> _postHeaders = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'ngrok-skip-browser-warning': 'true',
  };

  static String _shortBody(String body, {int max = 240}) {
    final clean = body.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (clean.length <= max) return clean;
    return '${clean.substring(0, max)}...';
  }

  static bool _looksLikeHtml(String body) {
    final b = body.trimLeft().toLowerCase();
    return b.startsWith('<!doctype html') ||
        b.startsWith('<html') ||
        b.startsWith('<!doctype');
  }

  static Map<String, dynamic> _decodeJsonObject(
    http.Response res, {
    required String endpointName,
  }) {
    final body = res.body;

    if (_looksLikeHtml(body)) {
      throw Exception(
        '$endpointName returned HTML instead of JSON.\n'
        'Status: ${res.statusCode}\n'
        'Content-Type: ${res.headers['content-type']}\n'
        'Check: ngrok HTTPS URL / route path / backend running.\n'
        'Body: ${_shortBody(body)}',
      );
    }

    dynamic decoded;
    try {
      decoded = jsonDecode(body);
    } catch (e) {
      throw Exception(
        '$endpointName JSON parse failed: $e\n'
        'Status: ${res.statusCode}\n'
        'Content-Type: ${res.headers['content-type']}\n'
        'Body: ${_shortBody(body)}',
      );
    }

    if (decoded is! Map<String, dynamic>) {
      throw Exception(
        '$endpointName expected JSON object but got ${decoded.runtimeType}. '
        'Body: ${_shortBody(body)}',
      );
    }

    return decoded;
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
      throw Exception('$endpointName request timeout');
    } catch (e) {
      throw Exception('$endpointName request failed: $e');
    }

    // ignore: avoid_print
    print('POST $uri -> ${res.statusCode} | ${res.headers['content-type']}');

    if (res.statusCode != 200) {
      if (_looksLikeHtml(res.body)) {
        throw Exception(
          '$endpointName failed (${res.statusCode}) and returned HTML, not JSON.\n'
          'Check ngrok URL, HTTPS, route path, and backend logs.\n'
          'Body: ${_shortBody(res.body)}',
        );
      }

      try {
        final errMap = jsonDecode(res.body) as Map<String, dynamic>;
        throw Exception(
            errMap['error']?.toString() ?? '$endpointName failed (${res.statusCode})');
      } catch (_) {
        throw Exception(
          '$endpointName failed: ${res.statusCode} ${_shortBody(res.body)}',
        );
      }
    }

    return res;
  }

  /// Call /api/recommend/recommend and return the raw JSON map.
  static Future<Map<String, dynamic>> _recommendRaw({
    required String query,
    int topK = 30,
  }) async {
    final res = await _post(
      '/api/recommend/recommend',
      endpointName: 'Recommendation',
      body: {
        'query': query,
        'top_k': topK,
      },
    );

    return _decodeJsonObject(res, endpointName: 'Recommendation');
  }

  /// Convenience for labour screens: returns the labour_recommendations list.
  static Future<List<Map<String, dynamic>>> recommendLabour({
    required String query,
    int topK = 40,
  }) async {
    final map = await _recommendRaw(query: query, topK: topK);
    final list = (map['labour_recommendations'] as List<dynamic>? ?? []);
    return list
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e as Map))
        .toList();
  }

  /// Convenience for equipment screens: returns the equipment_recommendations list.
  static Future<List<Map<String, dynamic>>> recommendEquipment({
    required String query,
    int topK = 40,
  }) async {
    final map = await _recommendRaw(query: query, topK: topK);
    final list = (map['equipment_recommendations'] as List<dynamic>? ?? []);
    return list
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e as Map))
        .toList();
  }
}

