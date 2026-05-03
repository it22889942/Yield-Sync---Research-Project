import 'package:flutter/material.dart';
import 'dart:async';
import 'config/firebase_config.dart';
import 'services/seasonal_notification_service.dart';
import 'ui/screens/labour/equipment_profile_screen.dart';
import 'ui/screens/labour/labour_profile_screen.dart';
import 'utils/app_theme.dart';
import 'services/app_routes.dart';

import 'ui/screens/splash_screen.dart';
import 'ui/screens/login_screen.dart';
import 'ui/screens/register_screen.dart';
import 'ui/screens/home_screen.dart';

import 'ui/screens/fertilizer_form_screen.dart';
import 'ui/screens/fertilizer_result_screen.dart';
import 'ui/screens/soil_quality_screen.dart';
import 'ui/screens/market_screen.dart';
import 'ui/screens/market_forecast_input_screen.dart';
import 'ui/screens/market_forecast_analytics_screen.dart';
import 'ui/screens/market_forecast_analytics_view_screen.dart';
import 'ui/screens/market_comparison_history_screen.dart';
import 'ui/screens/market_update_data_screen.dart';
import 'ui/screens/labor_hire_screen.dart';
import 'ui/screens/labor_list_screen.dart';
import 'ui/screens/labor_details_screen.dart';
import 'ui/screens/equipment_list_screen.dart';
import 'ui/screens/equipment_details_screen.dart';
import 'ui/screens/match_crop_screen.dart';

import 'ui/screens/admin_profile_screen.dart';
import 'ui/screens/profile_screen.dart';
import 'ui/screens/add_labour_post_screen.dart';
import 'ui/screens/add_equipment_post_screen.dart';
import 'ui/screens/my_labour_posts_screen.dart';
import 'ui/screens/my_equipment_posts_screen.dart';
import 'ui/screens/my_labour_bookings_screen.dart';
import 'ui/screens/my_equipment_bookings_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FirebaseConfig.init();
  runApp(const YieldSyncApp());

  unawaited(() async {
    await SeasonalNotificationService.instance.initialize();
    await SeasonalNotificationService.instance.notifyForTodayIfNeeded();
  }());
}

class YieldSyncApp extends StatelessWidget {
  const YieldSyncApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Yield Sync",
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      initialRoute: AppRoutes.splash,
      routes: {
        AppRoutes.splash: (_) => const SplashScreen(),
        AppRoutes.login: (_) => const LoginScreen(),
        AppRoutes.register: (_) => const RegisterScreen(),
        AppRoutes.home: (_) => const HomeScreen(),
        AppRoutes.fertilizerForm: (_) => const FertilizerFormScreen(),
        AppRoutes.soilQuality: (_) => const SoilQualityScreen(),
        AppRoutes.market: (_) => const MarketScreen(),
        AppRoutes.marketForecastInput: (_) => const MarketForecastInputScreen(),
        AppRoutes.marketForecastAnalyticsView: (_) =>
            const MarketForecastAnalyticsViewScreen(),
        AppRoutes.marketComparisonHistory: (_) =>
            const MarketComparisonHistoryScreen(),
        AppRoutes.marketDataUpdate: (_) => const MarketUpdateDataScreen(),
        AppRoutes.laborHire: (_) => const LaborHireScreen(),
        AppRoutes.laborList: (_) => const LaborListScreen(),
        AppRoutes.laborDetails: (_) => const LaborDetailsScreen(),
        AppRoutes.equipmentList: (_) => const EquipmentListScreen(),
        AppRoutes.equipmentDetails: (_) => const EquipmentDetailsScreen(),
        AppRoutes.matchCrop: (_) => const MatchCropScreen(),
        AppRoutes.adminProfile: (_) => const AdminProfileScreen(),
        AppRoutes.profile: (_) => const ProfileScreen(),
        AppRoutes.labourProfile: (_) => const LabourProfileScreen(),
        AppRoutes.equipmentProfile: (_) => const EquipmentProfileScreen(),
        AppRoutes.addLabourPost: (_) => const AddLabourPostScreen(),
        AppRoutes.addEquipmentPost: (_) => const AddEquipmentPostScreen(),
        AppRoutes.myLabourPosts: (_) => const MyLabourPostsScreen(),
        AppRoutes.myEquipmentPosts: (_) => const MyEquipmentPostsScreen(),
        AppRoutes.myLabourBookings: (_) => const MyLabourBookingsScreen(),
        AppRoutes.myEquipmentBookings: (_) => const MyEquipmentBookingsScreen(),
      },
    );
  }
}
