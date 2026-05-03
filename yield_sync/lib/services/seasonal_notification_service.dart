import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'seasonal_market_advisory_service.dart';

class SeasonalNotificationService {
  SeasonalNotificationService._();

  static final SeasonalNotificationService instance =
      SeasonalNotificationService._();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();
  final SeasonalMarketAdvisoryService _advisoryService =
      const SeasonalMarketAdvisoryService();

  bool _initialized = false;
  bool _enabled = false;

  Future<void> initialize() async {
    if (_initialized) return;

    // Local notifications are enabled only on mobile in this app.
    _enabled = !kIsWeb &&
        (defaultTargetPlatform == TargetPlatform.android ||
            defaultTargetPlatform == TargetPlatform.iOS);

    if (!_enabled) {
      _initialized = true;
      return;
    }

    try {
      const androidInit = AndroidInitializationSettings('@mipmap/ic_launcher');
      const iosInit = DarwinInitializationSettings();
      const initSettings =
          InitializationSettings(android: androidInit, iOS: iosInit);

      await _plugin.initialize(initSettings);

      if (defaultTargetPlatform == TargetPlatform.android) {
        final androidImpl =
            _plugin.resolvePlatformSpecificImplementation<
                AndroidFlutterLocalNotificationsPlugin>();
        await androidImpl?.requestNotificationsPermission();
      }
      if (defaultTargetPlatform == TargetPlatform.iOS) {
        final iosImpl = _plugin.resolvePlatformSpecificImplementation<
            IOSFlutterLocalNotificationsPlugin>();
        await iosImpl?.requestPermissions(
          alert: true,
          badge: true,
          sound: true,
        );
      }
    } catch (_) {
      _enabled = false;
    }

    _initialized = true;
  }

  Future<void> notifyForTodayIfNeeded() async {
    if (!_initialized || !_enabled) return;

    final today = DateTime.now();
    final advisories = _advisoryService.advisoriesForDate(today);
    if (advisories.isEmpty) return;

    final week = _advisoryService.weekOfMonth(today);
    final month = _advisoryService.monthName(today.month);
    final uniqueKey =
        '${today.year}-${today.month}-${today.day}-${advisories.map((e) => e.signal).join(',')}';

    try {
      final prefs = await SharedPreferences.getInstance();
      final lastKey = prefs.getString('seasonal_market_notified_key');
      if (lastKey == uniqueKey) return;

      final first = advisories.first;
      final extra =
          advisories.length > 1 ? ' (+${advisories.length - 1} more)' : '';
      final title = 'Market alert: Week $week of $month';
      final body = '${first.signal}: ${first.title} $extra';

      const androidDetails = AndroidNotificationDetails(
        'seasonal_market_alerts',
        'Seasonal Market Alerts',
        channelDescription: 'Festival and seasonal hold/sell alerts',
        importance: Importance.high,
        priority: Priority.high,
      );
      const iosDetails = DarwinNotificationDetails();
      const details =
          NotificationDetails(android: androidDetails, iOS: iosDetails);

      await _plugin.show(9001, title, body, details);
      await prefs.setString('seasonal_market_notified_key', uniqueKey);
    } catch (_) {
      // Do not block app flow if local notifications fail on a device.
    }
  }
}
