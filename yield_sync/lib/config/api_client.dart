class ApiClient {
  ApiClient._();

  /// ✅ Use HTTPS for ngrok
  static const String baseUrl = "https://49cf-212-104-231-106.ngrok-free.app";

  static Uri url(String path) {
    final cleanBase = baseUrl.endsWith("/")
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;

    final cleanPath = path.startsWith("/") ? path : "/$path";
    return Uri.parse("$cleanBase$cleanPath");
  }
}
