class ApiClient {
  ApiClient._();

  /// ✅ Use HTTPS for ngrok
  static const String baseUrl = "http://127.0.0.1:5003";

  static Uri url(String path) {
    final cleanBase = baseUrl.endsWith("/")
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;

    final cleanPath = path.startsWith("/") ? path : "/$path";
    return Uri.parse("$cleanBase$cleanPath");
  }
}
