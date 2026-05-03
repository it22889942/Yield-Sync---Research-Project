import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../utils/app_colors.dart';
import '../../data/crop_type_guides.dart';
import '../../services/app_routes.dart';
import '../../services/labor_api.dart';
import '../../services/seasonal_market_advisory_service.dart';
import '../../services/seasonal_notification_service.dart';
import '../../services/weather_service.dart';
import '../widgets/crop_type_guide_sheet.dart';
import 'equipment_rent_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  Future<void> _logout(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      barrierDismissible: true,
      builder: (_) => AlertDialog(
        title:
            const Text("Logout", style: TextStyle(fontWeight: FontWeight.w900)),
        content: const Text("Are you sure you want to logout?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Cancel",
                style: TextStyle(fontWeight: FontWeight.w800)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primary,
              foregroundColor: AppColors.darkGreen,
              elevation: 0,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text("Logout",
                style: TextStyle(fontWeight: FontWeight.w900)),
          ),
        ],
      ),
    );

    if (ok != true) return;

    await FirebaseAuth.instance.signOut();
    if (!context.mounted) return;

    Navigator.pushNamedAndRemoveUntil(context, AppRoutes.login, (r) => false);
  }

  void _goHome(BuildContext context) {
    final current = ModalRoute.of(context)?.settings.name;
    if (current == AppRoutes.home) return;

    Navigator.pushNamedAndRemoveUntil(context, AppRoutes.home, (r) => false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9FDF2),
      drawer: const _HomeDrawer(),
      body: const SafeArea(child: _HomeDashboardView()),

      bottomNavigationBar: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 10),
          child: Container(
            height: 64,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(999),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.08),
                  blurRadius: 18,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _BottomNavItem(
                  icon: Icons.home_rounded,
                  label: "Home",
                  active: true,
                  onTap: () => _goHome(context),
                ),
                _BottomNavItem(
                  icon: Icons.groups_rounded,
                  label: "Labour",
                  onTap: () => Navigator.pushNamed(
                    context,
                    AppRoutes.laborList,
                    arguments: LaborSearchArgs(
                      query: "",
                      location: "Kurunegala",
                      skill: null,
                    ),
                  ),
                ),
                _BottomNavItem(
                  icon: Icons.storefront_rounded,
                  label: "Market",
                  onTap: () => Navigator.pushNamed(context, AppRoutes.market),
                ),
                _BottomNavItem(
                  icon: Icons.person_rounded,
                  label: "Profile",
                  onTap: () => Navigator.pushNamed(context, AppRoutes.profile),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ===================== LEFT SIDEBAR DRAWER =====================
class _HomeDrawer extends StatelessWidget {
  const _HomeDrawer();

  Future<void> _logout(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      barrierDismissible: true,
      builder: (_) => AlertDialog(
        title:
            const Text("Logout", style: TextStyle(fontWeight: FontWeight.w900)),
        content: const Text("Are you sure you want to logout?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Cancel",
                style: TextStyle(fontWeight: FontWeight.w800)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primary,
              foregroundColor: AppColors.darkGreen,
              elevation: 0,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text("Logout",
                style: TextStyle(fontWeight: FontWeight.w900)),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await FirebaseAuth.instance.signOut();
    if (!context.mounted) return;
    Navigator.pushNamedAndRemoveUntil(context, AppRoutes.login, (r) => false);
  }

  @override
  Widget build(BuildContext context) {
    return Drawer(
      backgroundColor: const Color(0xFFF9FDF2),
      child: SafeArea(
        child: Column(
          children: [
            const _DrawerHeader(),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(12, 20, 12, 24),
                children: [
                  _DrawerSectionTitle("Your ads"),
                  const SizedBox(height: 8),
                  _DrawerTile(
                    icon: Icons.groups_rounded,
                    label: "Labour",
                    iconColor: const Color(0xFF15B77E),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, AppRoutes.myLabourPosts);
                    },
                  ),
                  const SizedBox(height: 6),
                  _DrawerTile(
                    icon: Icons.agriculture_rounded,
                    label: "Equipment",
                    iconColor: const Color(0xFF1C9BE8),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, AppRoutes.myEquipmentPosts);
                    },
                  ),
                  const SizedBox(height: 20),
                  _DrawerSectionTitle("Your bookings"),
                  const SizedBox(height: 8),
                  _DrawerTile(
                    icon: Icons.calendar_today_rounded,
                    label: "Labour",
                    iconColor: const Color(0xFF15B77E),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, AppRoutes.myLabourBookings);
                    },
                  ),
                  const SizedBox(height: 6),
                  _DrawerTile(
                    icon: Icons.agriculture_rounded,
                    label: "Equipment",
                    iconColor: const Color(0xFF1C9BE8),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, AppRoutes.myEquipmentBookings);
                    },
                  ),
                  const SizedBox(height: 20),
                  _DrawerSectionTitle("Account"),
                  const SizedBox(height: 8),
                  _DrawerTile(
                    icon: Icons.person_rounded,
                    label: "Profile",
                    iconColor: const Color(0xFF9B8CFF),
                    onTap: () {
                      Navigator.pop(context);
                      Navigator.pushNamed(context, AppRoutes.profile);
                    },
                  ),
                  const SizedBox(height: 6),
                  _DrawerTile(
                    icon: Icons.logout_rounded,
                    label: "Logout",
                    iconColor: AppColors.error,
                    onTap: () {
                      // Keep Scaffold context: after closing the drawer, the drawer's
                      // BuildContext is unmounted — dialogs/navigation would no-op or fail.
                      final scaffoldContext = Scaffold.of(context).context;
                      Navigator.pop(context);
                      _logout(scaffoldContext);
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DrawerHeader extends StatelessWidget {
  const _DrawerHeader();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 24),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(bottom: BorderSide(color: AppColors.border.withOpacity(0.8))),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(24),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: AppColors.border,
                width: 1,
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(10),
              child: Image.asset("assets/images/logo.png", fit: BoxFit.contain),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "YieldSync",
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark,
                    fontSize: 18,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  "Navigation",
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    color: AppColors.textDark.withOpacity(0.6),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DrawerSectionTitle extends StatelessWidget {
  final String title;

  const _DrawerSectionTitle(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4),
      child: Row(
        children: [
          Container(
            width: 4,
            height: 16,
            decoration: BoxDecoration(
              color: AppColors.primary,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            title.toUpperCase(),
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w700,
              color: AppColors.textDark.withOpacity(0.55),
              fontSize: 11,
              letterSpacing: 0.8,
            ),
          ),
        ],
      ),
    );
  }
}

class _DrawerTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color iconColor;
  final VoidCallback onTap;

  const _DrawerTile({
    required this.icon,
    required this.label,
    required this.onTap,
    Color? iconColor,
  }) : iconColor = iconColor ?? AppColors.darkGreen;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(16),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            child: Row(
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: iconColor.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(icon, color: iconColor, size: 22),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Text(
                    label,
                    style: GoogleFonts.inter(
                      fontWeight: FontWeight.w700,
                      color: AppColors.textDark.withOpacity(0.95),
                      fontSize: 15.2,
                    ),
                  ),
                ),
                Icon(
                  Icons.chevron_right_rounded,
                  size: 22,
                  color: AppColors.textDark.withOpacity(0.35),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ===================== HOME DASHBOARD =====================
class _HomeDashboardView extends StatefulWidget {
  const _HomeDashboardView();

  @override
  State<_HomeDashboardView> createState() => _HomeDashboardViewState();
}

class _HomeDashboardViewState extends State<_HomeDashboardView> {
  final SeasonalMarketAdvisoryService _advisoryService =
      const SeasonalMarketAdvisoryService();
  List<SeasonalMarketAdvisory> _todayAdvisories =
      <SeasonalMarketAdvisory>[];
  late DateTime _today;
  late Future<WeatherInfo> _weatherFuture;

  /// Selected chip under "Crop Types" (crop name; empty until user taps one).
  String _selectedCropType = "";

  /// Core Features: horizontal strip vs 2-column grid ("View all" / "View less").
  bool _coreFeaturesExpanded = false;

  @override
  void initState() {
    super.initState();
    _today = DateTime.now();
    _todayAdvisories = _advisoryService.advisoriesForDate(_today);
    SeasonalNotificationService.instance.notifyForTodayIfNeeded();
    _weatherFuture = WeatherService.fetchByLatLon(7.2906, 80.6337);
  }

  static String _formatHeaderDate(DateTime d) {
    const weekdays = [
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday",
    ];
    const months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];
    return "${weekdays[d.weekday - 1]}, ${d.day.toString().padLeft(2, "0")} "
        "${months[d.month - 1]} ${d.year}";
  }

  static String _greetingForHour(int hour) {
    if (hour < 12) return "Hello, Good Morning";
    if (hour < 17) return "Hello, Good Afternoon";
    return "Hello, Good Evening";
  }

  Future<void> _onDashboardRefresh() async {
    setState(() {
      _today = DateTime.now();
      _todayAdvisories = _advisoryService.advisoriesForDate(_today);
      _weatherFuture = WeatherService.fetchByLatLon(7.2906, 80.6337);
    });
    try {
      await _weatherFuture;
    } catch (_) {
      if (mounted) setState(() {});
    }
  }

  void _onCropTypeChipTap(String label) {
    setState(() => _selectedCropType = label);
    final guide = kCropTypeGuides[label];
    if (guide != null && mounted) {
      showCropTypeGuideSheet(context, guide);
    }
  }

  Future<void> _logout(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      barrierDismissible: true,
      builder: (_) => AlertDialog(
        title:
            const Text("Logout", style: TextStyle(fontWeight: FontWeight.w900)),
        content: const Text("Are you sure you want to logout?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text("Cancel",
                style: TextStyle(fontWeight: FontWeight.w800)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primary,
              foregroundColor: AppColors.darkGreen,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            child: const Text("Logout",
                style: TextStyle(fontWeight: FontWeight.w900)),
          ),
        ],
      ),
    );

    if (ok != true) return;

    await FirebaseAuth.instance.signOut();
    if (!context.mounted) return;

    Navigator.pushNamedAndRemoveUntil(context, AppRoutes.login, (r) => false);
  }

  @override
  Widget build(BuildContext context) {
    final actions = <_ActionItem>[
      _ActionItem(
        title: "Fertiliser Suggest",
        subtitle: "Best NPK plan",
        icon: Icons.science_rounded,
        iconBg: const Color(0xFF35D6A6),
        route: AppRoutes.fertilizerForm,
      ),
      _ActionItem(
        title: "Hire Labour",
        subtitle: "Skilled workers",
        icon: Icons.groups_rounded,
        iconBg: const Color(0xFF15B77E),
        route: AppRoutes.laborList,
        routeArguments: LaborSearchArgs(
          query: "",
          location: "Kurunegala",
          skill: null,
        ),
      ),
      _ActionItem(
        title: "Rent Equipment",
        subtitle: "Tractors and more",
        icon: Icons.agriculture_rounded,
        iconBg: const Color(0xFF1C9BE8),
        route: AppRoutes.equipmentList,
        routeArguments: EquipmentSearchArgs(
          query: "",
          location: "Kurunegala",
          type: "",
        ),
      ),
      _ActionItem(
        title: "Match Crop",
        subtitle: "AI crop guidance",
        icon: Icons.spa_rounded,
        iconBg: const Color(0xFF9B8CFF),
        route: AppRoutes.matchCrop,
      ),
      _ActionItem(
        title: "Market",
        subtitle: "Daily prices",
        icon: Icons.storefront_rounded,
        iconBg: const Color(0xFFFFB74D),
        route: AppRoutes.market,
      ),
      _ActionItem(
        title: "Soil Quality",
        subtitle: "Live soil & environment",
        icon: Icons.grass_rounded,
        iconBg: const Color(0xFF2BB3D1),
        route: AppRoutes.soilQuality,
      ),
    ];
    final offers = [...actions];
    offers.sort((a, b) {
      if (a.title == "Hire Labour") return -1;
      if (b.title == "Hire Labour") return 1;
      return 0;
    });

    // Best Offers: Soil Quality is 3rd; then Rent, Match, Market (6 cards total).
    final soilOffer =
        actions.firstWhere((a) => a.title == "Soil Quality");
    final withoutSoil =
        offers.where((a) => a.title != "Soil Quality").toList();
    final bestOffersForRow = [
      ...withoutSoil.take(2),
      soilOffer,
      ...withoutSoil.skip(2), // Rent Equipment, Match Crop, Market
    ];

    final now = DateTime.now();
    return RefreshIndicator(
      onRefresh: _onDashboardRefresh,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(
          parent: BouncingScrollPhysics(),
        ),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: AppColors.border),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.08),
                    blurRadius: 26,
                    offset: const Offset(0, 16),
                  ),
                ],
              ),
              child: Stack(
                children: [
                  Positioned(
                    right: -26,
                    top: -20,
                    child: Container(
                      width: 120,
                      height: 120,
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.14),
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                  Positioned(
                    right: 40,
                    bottom: -30,
                    child: Container(
                      width: 92,
                      height: 92,
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.08),
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(
                    children: [
                      IconButton(
                        onPressed: () => Scaffold.of(context).openDrawer(),
                        icon: const Icon(Icons.menu_rounded,
                            color: AppColors.darkGreen),
                        style: IconButton.styleFrom(
                          backgroundColor: AppColors.primary.withOpacity(0.18),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                      const Spacer(),
                      _HeaderIconBtn(
                        icon: Icons.person_rounded,
                        tooltip: "Profile",
                        onTap: () => Navigator.pushNamed(context, AppRoutes.profile),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    _greetingForHour(now.hour),
                    style: GoogleFonts.poppins(
                      color: AppColors.textDark,
                      fontWeight: FontWeight.w600,
                      fontSize: 24,
                    ),
                  ),
                  Text(
                    _formatHeaderDate(now),
                    style: GoogleFonts.inter(
                      color: AppColors.textDark.withOpacity(0.62),
                      fontWeight: FontWeight.w500,
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 12),
                  FutureBuilder<WeatherInfo>(
                    future: _weatherFuture,
                    builder: (context, snapshot) {
                      if (!snapshot.hasData) {
                        return Container(
                          height: 118,
                          decoration: BoxDecoration(
                            color: AppColors.surface,
                            borderRadius: BorderRadius.circular(18),
                            border: Border.all(color: AppColors.border),
                          ),
                          alignment: Alignment.center,
                          child: SizedBox(
                            width: 22,
                            height: 22,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: AppColors.darkGreen.withOpacity(0.85),
                            ),
                          ),
                        );
                      }
                      final w = snapshot.data!;
                      return Container(
                        padding: const EdgeInsets.fromLTRB(12, 12, 12, 10),
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Column(
                          children: [
                            Row(
                              children: [
                                Icon(Icons.location_on_rounded,
                                    color: AppColors.darkGreen, size: 16),
                                const SizedBox(width: 5),
                                Text(
                                  w.city,
                                  style: GoogleFonts.inter(
                                    color: AppColors.textDark,
                                    fontSize: 13.2,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                const Spacer(),
                                Text(
                                  w.condition,
                                  style: GoogleFonts.inter(
                                    color: AppColors.textDark.withOpacity(0.78),
                                    fontSize: 12.4,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Row(
                              children: [
                                Text(
                                  "${w.tempC.round()}°C",
                                  style: GoogleFonts.poppins(
                                    color: AppColors.darkGreen,
                                    fontSize: 42,
                                    height: 1,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                const Spacer(),
                                Icon(Icons.cloud_rounded,
                                    color: AppColors.primary.withOpacity(0.95),
                                    size: 34),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Row(
                              children: [
                                Expanded(child: _HeaderMetric(title: "Humidity", value: "${w.humidity.round()}%")),
                                Expanded(child: _HeaderMetric(title: "Rain", value: "${w.rainfallMm.toStringAsFixed(1)} mm")),
                                Expanded(
                                  child: _HeaderMetric(
                                    title: "Wind",
                                    value:
                                        "${w.windSpeedKmh.round()} km/h",
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ],
              ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                "Crop Types",
                style: GoogleFonts.poppins(
                  fontSize: 21,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textDark,
                ),
              ),
            ),
            const SizedBox(height: 6),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _FieldChip(
                    label: "Rice",
                    active: _selectedCropType == "Rice",
                    onTap: () => _onCropTypeChipTap("Rice"),
                  ),
                  _FieldChip(
                    label: "Radish",
                    active: _selectedCropType == "Radish",
                    onTap: () => _onCropTypeChipTap("Radish"),
                  ),
                  _FieldChip(
                    label: "Beetroot",
                    active: _selectedCropType == "Beetroot",
                    onTap: () => _onCropTypeChipTap("Beetroot"),
                  ),
                  _FieldChip(
                    label: "Red Onion",
                    active: _selectedCropType == "Red Onion",
                    onTap: () => _onCropTypeChipTap("Red Onion"),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            const SizedBox(height: 8),
            Row(
              children: [
                Text(
                  "Core Features",
                  style: GoogleFonts.poppins(
                    fontSize: 21,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark,
                  ),
                ),
                const Spacer(),
                TextButton(
                  onPressed: () {
                    setState(() {
                      _coreFeaturesExpanded = !_coreFeaturesExpanded;
                    });
                  },
                  child: Text(
                    _coreFeaturesExpanded ? "View less" : "View all",
                    style: GoogleFonts.inter(
                      fontWeight: FontWeight.w600,
                      color: AppColors.darkGreen,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            if (!_coreFeaturesExpanded)
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: bestOffersForRow.map((a) {
                    return Padding(
                      padding: const EdgeInsets.only(right: 10),
                      child: _OfferTile(
                        item: a,
                        expandWidth: false,
                        onTap: () => Navigator.pushNamed(
                          context,
                          a.route!,
                          arguments: a.routeArguments,
                        ),
                      ),
                    );
                  }).toList(),
                ),
              )
            else
              LayoutBuilder(
                builder: (context, constraints) {
                  const spacing = 10.0;
                  final maxW = constraints.maxWidth;
                  final tileW = maxW > spacing
                      ? (maxW - spacing) / 2
                      : maxW;
                  return Wrap(
                    spacing: spacing,
                    runSpacing: spacing,
                    alignment: WrapAlignment.start,
                    children: bestOffersForRow.map((a) {
                      return SizedBox(
                        width: tileW,
                        child: _OfferTile(
                          item: a,
                          expandWidth: true,
                          onTap: () => Navigator.pushNamed(
                            context,
                            a.route!,
                            arguments: a.routeArguments,
                          ),
                        ),
                      );
                    }).toList(),
                  );
                },
              ),
            const SizedBox(height: 14),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                children: [
                  Container(
                    width: 34,
                    height: 34,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(Icons.tips_and_updates_rounded,
                        color: AppColors.darkGreen, size: 20),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      "Tip: Keep your market and weather insights updated before making crop and equipment decisions.",
                      style: GoogleFonts.inter(
                        fontSize: 12.4,
                        height: 1.35,
                        fontWeight: FontWeight.w500,
                        color: AppColors.textDark.withOpacity(0.72),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            _SeasonalMarketAlertCard(
              today: _today,
              service: _advisoryService,
              advisories: _todayAdvisories,
            ),
          ],
        ),
      ),
    ),
    );
  }
}

class _ModernKpiCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;

  const _ModernKpiCard({
    required this.title,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.18),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: AppColors.darkGreen, size: 18),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  value,
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                    color: AppColors.textDark,
                  ),
                ),
                Text(
                  title,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w500,
                    fontSize: 11.5,
                    color: AppColors.textDark.withOpacity(0.64),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _WeatherOverviewCard extends StatelessWidget {
  final WeatherInfo info;

  const _WeatherOverviewCard({required this.info});

  String _weekdayName(int weekday) {
    const names = [
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday"
    ];
    return names[(weekday - 1).clamp(0, 6)];
  }

  String _monthName(int month) {
    const names = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec"
    ];
    return names[(month - 1).clamp(0, 11)];
  }

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now();
    final dateLabel = "${_weekdayName(now.weekday)}, ${now.day.toString().padLeft(2, '0')} ${_monthName(now.month)} ${now.year}";

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.location_on_rounded,
                  color: AppColors.darkGreen.withOpacity(0.82), size: 18),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  info.city,
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w700,
                    fontSize: 15.2,
                    color: AppColors.textDark,
                  ),
                ),
              ),
              Text(
                info.condition,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 12.4,
                  color: AppColors.textDark.withOpacity(0.64),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Text(
                "${info.tempC.round()}°C",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  fontSize: 42,
                  height: 1,
                  color: AppColors.textDark,
                ),
              ),
              const Spacer(),
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(Icons.cloud_rounded,
                    color: AppColors.darkGreen, size: 32),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _WeatherMetric(
                  title: "Humidity",
                  value: "${info.humidity.round()}%",
                ),
              ),
              Expanded(
                child: _WeatherMetric(
                  title: "Rain",
                  value: "${info.rainfallMm.toStringAsFixed(1)} mm",
                ),
              ),
              const Expanded(
                child: _WeatherMetric(
                  title: "Pressure",
                  value: "1008 hPa",
                ),
              ),
              const Expanded(
                child: _WeatherMetric(
                  title: "Wind",
                  value: "18 km/h",
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Divider(color: AppColors.border.withOpacity(0.8), height: 1),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: Row(
                  children: [
                    Container(
                      width: 28,
                      height: 28,
                      decoration: BoxDecoration(
                        color: const Color(0xFFFFF4DA),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(
                        Icons.wb_sunny_outlined,
                        size: 16,
                        color: Color(0xFFE5A100),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      "5:25 am\nSunrise",
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        height: 1.3,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textDark.withOpacity(0.72),
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                width: 1,
                height: 30,
                color: AppColors.border.withOpacity(0.9),
              ),
              Expanded(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    Text(
                      "8:04 pm\nSunset",
                      textAlign: TextAlign.right,
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        height: 1.3,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textDark.withOpacity(0.72),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      width: 28,
                      height: 28,
                      decoration: BoxDecoration(
                        color: const Color(0xFFE9EEFF),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(
                        Icons.nights_stay_outlined,
                        size: 16,
                        color: Color(0xFF5667B8),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _WeatherMetric extends StatelessWidget {
  final String title;
  final String value;

  const _WeatherMetric({
    required this.title,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.inter(
              fontSize: 11.4,
              fontWeight: FontWeight.w500,
              color: AppColors.textDark.withOpacity(0.58),
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: GoogleFonts.poppins(
              fontSize: 13.4,
              fontWeight: FontWeight.w700,
              color: AppColors.textDark,
            ),
          ),
        ],
      ),
    );
  }
}

class _WeatherCardSkeleton extends StatelessWidget {
  const _WeatherCardSkeleton();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 170,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
      ),
      alignment: Alignment.center,
      child: const CircularProgressIndicator(strokeWidth: 2),
    );
  }
}

class _PrimaryDashboardAction extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool outlined;
  final VoidCallback onTap;

  const _PrimaryDashboardAction({
    required this.label,
    required this.icon,
    required this.onTap,
    this.outlined = false,
  });

  @override
  Widget build(BuildContext context) {
    final child = Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, size: 18),
        const SizedBox(width: 8),
        Flexible(
          child: Text(
            label,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.poppins(
              fontWeight: FontWeight.w700,
              fontSize: 13.5,
            ),
          ),
        ),
      ],
    );

    if (outlined) {
      return SizedBox(
        height: 48,
        child: OutlinedButton(
          onPressed: onTap,
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.darkGreen,
            side: const BorderSide(color: AppColors.border),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          ),
          child: child,
        ),
      );
    }

    return SizedBox(
      height: 48,
      child: ElevatedButton(
        onPressed: onTap,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: AppColors.darkGreen,
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
        child: child,
      ),
    );
  }
}

class _ServiceListCard extends StatelessWidget {
  final _ActionItem item;
  final VoidCallback onTap;

  const _ServiceListCard({required this.item, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Ink(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
          ),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: item.iconBg,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(item.icon, color: Colors.white),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.title,
                        style: GoogleFonts.poppins(
                          fontWeight: FontWeight.w700,
                          fontSize: 15.5,
                          color: AppColors.textDark,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        item.subtitle,
                        style: GoogleFonts.inter(
                          fontWeight: FontWeight.w500,
                          fontSize: 12.8,
                          color: AppColors.textDark.withOpacity(0.62),
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.chevron_right_rounded,
                  color: AppColors.textDark.withOpacity(0.42),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _MiniCategoryCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final VoidCallback onTap;

  const _MiniCategoryCard({
    required this.icon,
    required this.title,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Ink(
          padding: const EdgeInsets.fromLTRB(8, 12, 8, 10),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.025),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            children: [
              Container(
                width: 30,
                height: 30,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(9),
                ),
                child: Icon(icon, size: 17, color: AppColors.darkGreen),
              ),
              const SizedBox(height: 6),
              Text(
                title,
                textAlign: TextAlign.center,
                style: GoogleFonts.inter(
                  fontSize: 11.3,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textDark.withOpacity(0.9),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _OfferTile extends StatelessWidget {
  final _ActionItem item;
  final VoidCallback onTap;
  /// In a 2-column grid, fill cell width; otherwise fixed [172] for horizontal list.
  final bool expandWidth;

  const _OfferTile({
    required this.item,
    required this.onTap,
    this.expandWidth = false,
  });

  String? _imageForTitle(String title) {
    final t = title.toLowerCase();
    if (t.contains("labour")) return "assets/images/labour.jpeg";
    if (t.contains("market") || t.contains("price")) {
      return "assets/images/price.jpeg";
    }
    if (t.contains("fertilizer") || t.contains("fertiliser")) {
      return "assets/images/fertilizer.jpeg";
    }
    if (t.contains("crop") || t.contains("match")) {
      return "assets/images/crop.jpg";
    }
    if (t.contains("equipment") || t.contains("tractor")) {
      return "assets/images/tractor.jpeg";
    }
    if (t.contains("soil")) {
      return "assets/images/soilbanner.jpg";
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final inner = Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap,
          child: Ink(
            padding: const EdgeInsets.fromLTRB(10, 10, 10, 10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.04),
                  blurRadius: 14,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  height: 112,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        item.iconBg.withOpacity(0.92),
                        item.iconBg.withOpacity(0.68),
                      ],
                    ),
                  ),
                  child: Stack(
                    children: [
                      Positioned.fill(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: _imageForTitle(item.title) != null
                              ? Image.asset(
                                  _imageForTitle(item.title)!,
                                  fit: BoxFit.cover,
                                )
                              : Container(
                                  color: Colors.transparent,
                                  child: Center(
                                    child: Icon(item.icon,
                                        color: Colors.white, size: 30),
                                  ),
                                ),
                        ),
                      ),
                      Positioned.fill(
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12),
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [
                                Colors.black.withOpacity(0.12),
                                Colors.black.withOpacity(0.28),
                              ],
                            ),
                          ),
                        ),
                      ),
                      Positioned(
                        top: 10,
                        left: 10,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.24),
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            "Featured",
                            style: GoogleFonts.inter(
                              color: Colors.white,
                              fontSize: 10.6,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  item.title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.poppins(
                    fontSize: 14.5,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark,
                  ),
                ),
                Text(
                  item.subtitle,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    height: 1.25,
                    color: AppColors.textDark.withOpacity(0.62),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.18),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        "Explore",
                        style: GoogleFonts.inter(
                          fontSize: 11.6,
                          fontWeight: FontWeight.w700,
                          color: AppColors.darkGreen,
                        ),
                      ),
                    ),
                    const Spacer(),
                    Icon(Icons.arrow_forward_rounded,
                        size: 16, color: AppColors.darkGreen.withOpacity(0.9)),
                  ],
                ),
              ],
            ),
          ),
        ),
    );
    if (expandWidth) {
      return SizedBox(width: double.infinity, child: inner);
    }
    return SizedBox(width: 172, child: inner);
  }
}

class _SeasonalMarketAlertCard extends StatelessWidget {
  final DateTime today;
  final SeasonalMarketAdvisoryService service;
  final List<SeasonalMarketAdvisory> advisories;

  const _SeasonalMarketAlertCard({
    required this.today,
    required this.service,
    required this.advisories,
  });

  @override
  Widget build(BuildContext context) {
    final week = service.weekOfMonth(today);
    final monthName = service.monthName(today.month);
    final season = service.seasonName(today);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 34,
                height: 34,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.20),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(Icons.notifications_active_rounded,
                    color: AppColors.darkGreen, size: 20),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text(
                  "Seasonal Market Alert",
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark,
                    fontSize: 15.2,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            "$season - $monthName Week $week",
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              color: AppColors.textDark.withOpacity(0.65),
              fontSize: 12.8,
            ),
          ),
          const SizedBox(height: 10),
          if (advisories.isEmpty)
            Text(
              "No HOLD/SELL alert for this week. Monitor market trend and check next week.",
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w500,
                height: 1.35,
                color: AppColors.textDark.withOpacity(0.70),
              ),
            )
          else
            ...advisories.map((item) => _SeasonalSignalTile(item: item)),
        ],
      ),
    );
  }
}

class _SeasonalSignalTile extends StatelessWidget {
  final SeasonalMarketAdvisory item;

  const _SeasonalSignalTile({required this.item});

  @override
  Widget build(BuildContext context) {
    final isSell = item.signal.toUpperCase() == 'SELL';
    final chipColor = isSell ? const Color(0xFFE8F7EB) : const Color(0xFFFFF4E5);
    final chipTextColor = isSell ? const Color(0xFF176A2A) : const Color(0xFF8C5A11);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: chipColor,
              borderRadius: BorderRadius.circular(999),
            ),
            child: Text(
              item.signal,
              style: TextStyle(
                fontWeight: FontWeight.w900,
                color: chipTextColor,
                fontSize: 11.8,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            item.title,
            style: const TextStyle(
              fontWeight: FontWeight.w800,
              color: AppColors.textDark,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            item.detail,
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.textDark.withOpacity(0.65),
            ),
          ),
        ],
      ),
    );
  }
}

// ===================== SMALL WIDGETS =====================

class _HeaderIconBtn extends StatelessWidget {
  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;

  const _HeaderIconBtn({
    required this.icon,
    required this.tooltip,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: InkWell(
        borderRadius: BorderRadius.circular(999),
        onTap: onTap,
        child: CircleAvatar(
          radius: 18,
          backgroundColor: AppColors.primary.withOpacity(0.22),
          child: Icon(icon, color: AppColors.darkGreen),
        ),
      ),
    );
  }
}

class _BottomNavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool active;

  const _BottomNavItem({
    required this.icon,
    required this.label,
    required this.onTap,
    this.active = false,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(999),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: active ? AppColors.primary.withOpacity(0.2) : Colors.transparent,
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          children: [
            Icon(icon,
                size: 19, color: active ? AppColors.darkGreen : AppColors.textDark),
            if (active) ...[
              const SizedBox(width: 6),
              Text(
                label,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w700,
                  color: AppColors.darkGreen,
                  fontSize: 12.5,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _HeaderMetric extends StatelessWidget {
  final String title;
  final String value;

  const _HeaderMetric({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.inter(
            fontSize: 11,
            fontWeight: FontWeight.w500,
            color: AppColors.textDark.withOpacity(0.52),
          ),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: GoogleFonts.inter(
            fontSize: 12.8,
            fontWeight: FontWeight.w700,
            color: AppColors.textDark,
          ),
        ),
      ],
    );
  }
}

class _FieldChip extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _FieldChip({
    required this.label,
    this.active = false,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: Material(
        color: active ? AppColors.primary.withOpacity(0.25) : Colors.white,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.border),
            ),
            child: Text(
              label,
              style: GoogleFonts.inter(
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
                color: AppColors.textDark.withOpacity(0.9),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _PillChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _PillChip({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 10),
      child: Material(
        color: AppColors.surface,
        shape: StadiumBorder(side: BorderSide(color: AppColors.border)),
        child: InkWell(
          customBorder: const StadiumBorder(),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 16, color: AppColors.darkGreen),
                const SizedBox(width: 8),
                Text(
                  label,
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark.withOpacity(0.92),
                    fontSize: 12.6,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;

  const _StatCard({
    required this.value,
    required this.label,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: AppColors.darkGreen),
          const SizedBox(height: 6),
          Text(
            value,
            style: GoogleFonts.poppins(
              color: AppColors.textDark,
              fontWeight: FontWeight.w700,
              fontSize: 15.6,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            label,
            style: GoogleFonts.inter(
              color: AppColors.textDark.withOpacity(0.62),
              fontWeight: FontWeight.w600,
              fontSize: 11.2,
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionItem {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color iconBg;
  final String? route;
  final Object? routeArguments;

  const _ActionItem({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.iconBg,
    this.route,
    this.routeArguments,
  });
}

class _ActionCard extends StatelessWidget {
  final _ActionItem item;
  final VoidCallback onTap;

  const _ActionCard({required this.item, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(20),
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
            child: Stack(
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: item.iconBg,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Icon(item.icon, color: Colors.white, size: 24),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      item.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.poppins(
                        fontWeight: FontWeight.w700,
                        color: AppColors.textDark,
                        fontSize: 14.4,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      item.subtitle,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.inter(
                        fontWeight: FontWeight.w500,
                        color: AppColors.textDark.withOpacity(0.6),
                        fontSize: 12.4,
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
                Positioned(
                  top: 0,
                  right: 0,
                  child: Container(
                    width: 32,
                    height: 32,
                    decoration: BoxDecoration(
                      color: AppColors.textDark.withOpacity(0.09),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      Icons.chevron_right_rounded,
                      size: 20,
                      color: AppColors.textDark.withOpacity(0.7),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
