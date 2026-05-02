import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../../services/seasonal_market_advisory_service.dart';
import '../../services/seasonal_notification_service.dart';

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
      backgroundColor: AppColors.surface,
      body: const SafeArea(child: _HomeDashboardView()),

      // ✅ Bottom home pill (kept, slightly improved)
      bottomNavigationBar: SafeArea(
        top: false,
        child: SizedBox(
          height: 68,
          child: BottomAppBar(
            color: Colors.white,
            elevation: 0,
            padding: EdgeInsets.zero,
            child: Center(
              child: Material(
                color: AppColors.primary.withOpacity(0.12),
                shape: StadiumBorder(
                  side: BorderSide(color: AppColors.border),
                ),
                child: InkWell(
                  customBorder: const StadiumBorder(),
                  onTap: () => _goHome(context),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 18, vertical: 11),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.home_rounded,
                            color: AppColors.darkGreen),
                        const SizedBox(width: 8),
                        Text(
                          "Home",
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark.withOpacity(0.92),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
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

  @override
  void initState() {
    super.initState();
    _today = DateTime.now();
    _todayAdvisories = _advisoryService.advisoriesForDate(_today);
    SeasonalNotificationService.instance.notifyForTodayIfNeeded();
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
        bg: const Color(0x1466D9AC),
        iconBg: const Color(0x2666D9AC),
        route: AppRoutes.fertilizerForm,
      ),
      _ActionItem(
        title: "Hire Labour",
        subtitle: "Skilled workers",
        icon: Icons.groups_rounded,
        bg: const Color(0x141C9BE8),
        iconBg: const Color(0x261C9BE8),
        route: AppRoutes.laborHire,
      ),
      _ActionItem(
        title: "Rent Equipment",
        subtitle: "Tools & tractors",
        icon: Icons.agriculture_rounded,
        bg: const Color(0x14FFB74D),
        iconBg: const Color(0x26FFB74D),
        route: AppRoutes.equipmentRent,
      ),
      _ActionItem(
        title: "Match Crop",
        subtitle: "Smart crop pick",
        icon: Icons.spa_rounded,
        bg: const Color(0x149B8CFF),
        iconBg: const Color(0x269B8CFF),
        route: AppRoutes.matchCrop,
      ),
      _ActionItem(
        title: "Market",
        subtitle: "Live prices",
        icon: Icons.storefront_rounded,
        bg: const Color(0x1466D9AC),
        iconBg: const Color(0x2666D9AC),
        route: AppRoutes.market,
      ),
      _ActionItem(
        title: "Soil Quality",
        subtitle: "pH / EC / NPK",
        icon: Icons.grass_rounded,
        bg: const Color(0x141C9BE8),
        iconBg: const Color(0x261C9BE8),
        route: AppRoutes.soilQuality,
      ),
    ];

    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      child: Column(
        children: [
          // ===================== HERO HEADER (upgraded) =====================
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
            decoration: const BoxDecoration(
              gradient: AppColors.heroGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(28),
                bottomRight: Radius.circular(28),
              ),
            ),
            child: Stack(
              children: [
                const Positioned(
                  right: -12,
                  top: 26,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.agriculture_rounded,
                        size: 150, color: Colors.white),
                  ),
                ),
                const Positioned(
                  right: 18,
                  bottom: -14,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.park_rounded,
                        size: 130, color: Colors.white),
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Top Row (Brand + profile + logout)
                    Row(
                      children: [
                        Container(
                          width: 44,
                          height: 44,
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.10),
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(
                                color: Colors.white.withOpacity(0.14)),
                          ),
                          child: const Icon(Icons.eco_rounded,
                              color: AppColors.primary),
                        ),
                        const SizedBox(width: 10),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "YieldSync",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.96),
                                fontWeight: FontWeight.w900,
                                fontSize: 15.8,
                              ),
                            ),
                            Text(
                              "SRI LANKA",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.72),
                                fontWeight: FontWeight.w700,
                                fontSize: 11.5,
                                letterSpacing: 1.0,
                              ),
                            ),
                          ],
                        ),
                        const Spacer(),
                        Row(
                          children: [
                            _HeaderIconBtn(
                              icon: Icons.person_rounded,
                              tooltip: "Profile",
                              onTap: () => Navigator.pushNamed(
                                  context, AppRoutes.profile),
                            ),
                            const SizedBox(width: 10),
                            _HeaderIconBtn(
                              icon: Icons.logout_rounded,
                              tooltip: "Logout",
                              onTap: () => _logout(context),
                            ),
                          ],
                        ),
                      ],
                    ),

                    const SizedBox(height: 14),

                    // Badge
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 7),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(999),
                        border:
                            Border.all(color: Colors.white.withOpacity(0.14)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.bolt_rounded,
                              color: AppColors.primary, size: 16),
                          const SizedBox(width: 8),
                          Text(
                            "Smart farming • Hire • Rent • Forecast",
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.90),
                              fontWeight: FontWeight.w800,
                              fontSize: 12.2,
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 14),

                    Text(
                      "Everything for\nYour Farm",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.98),
                        fontWeight: FontWeight.w900,
                        fontSize: 28.5,
                        height: 1.10,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      "Find labour, rent equipment and get\nsmart suggestions across Sri Lanka",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontWeight: FontWeight.w600,
                        fontSize: 13.6,
                        height: 1.25,
                      ),
                    ),

                    const SizedBox(height: 14),

                    // ✅ Updated stats: Equipment 500+ | Labour 500+ | Districts 25 | Rating 4.8
                    const Row(
                      children: [
                        Expanded(
                          child: _StatCard(
                            value: "500+",
                            label: "Equipment",
                            icon: Icons.agriculture_rounded,
                          ),
                        ),
                        SizedBox(width: 10),
                        Expanded(
                          child: _StatCard(
                            value: "500+",
                            label: "Labour",
                            icon: Icons.groups_rounded,
                          ),
                        ),
                        SizedBox(width: 10),
                        Expanded(
                          child: _StatCard(
                            value: "25",
                            label: "Districts",
                            icon: Icons.map_rounded,
                          ),
                        ),
                        SizedBox(width: 10),
                        Expanded(
                          child: _StatCard(
                            value: "4.8",
                            label: "Rating",
                            icon: Icons.star_rounded,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 14),

                    // ✅ Modern CTA Row (super interactive)
                    Row(
                      children: [
                        Expanded(
                          child: SizedBox(
                            height: 46,
                            child: ElevatedButton.icon(
                              onPressed: () => Navigator.pushNamed(
                                  context, AppRoutes.market),
                              icon: const Icon(Icons.trending_up_rounded),
                              label: const Text(
                                "Market Prices",
                                style: TextStyle(fontWeight: FontWeight.w900),
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.white,
                                foregroundColor: AppColors.darkGreen,
                                elevation: 0,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        SizedBox(
                          height: 46,
                          width: 54,
                          child: OutlinedButton(
                            onPressed: () =>
                                Navigator.pushNamed(context, AppRoutes.profile),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: Colors.white,
                              side: BorderSide(
                                  color: Colors.white.withOpacity(0.25)),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            child: const Icon(Icons.settings_rounded),
                          ),
                        ),
                      ],
                    ),
                  ],
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
          const SizedBox(height: 14),

          // ===================== QUICK ACTIONS (upgraded header + chips) =====================
          Container(
            width: double.infinity,
            margin: const EdgeInsets.symmetric(horizontal: 16),
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 18),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.06),
                  blurRadius: 22,
                  offset: const Offset(0, 14),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 38,
                      height: 38,
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.18),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Icon(Icons.flash_on_rounded,
                          color: AppColors.darkGreen),
                    ),
                    const SizedBox(width: 10),
                    const Expanded(
                      child: Text(
                        "Quick Actions",
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w900,
                          color: AppColors.textDark,
                        ),
                      ),
                    ),
                    Text(
                      "Tap any card",
                      style: TextStyle(
                        fontWeight: FontWeight.w800,
                        color: AppColors.textDark.withOpacity(0.55),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                // ✅ quick filter chips (no backend)
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  physics: const BouncingScrollPhysics(),
                  child: Row(
                    children: [
                      _PillChip(
                        icon: Icons.science_rounded,
                        label: "Suggestions",
                        onTap: () => Navigator.pushNamed(
                            context, AppRoutes.fertilizerForm),
                      ),
                      _PillChip(
                        icon: Icons.groups_rounded,
                        label: "Labour",
                        onTap: () =>
                            Navigator.pushNamed(context, AppRoutes.laborHire),
                      ),
                      _PillChip(
                        icon: Icons.agriculture_rounded,
                        label: "Equipment",
                        onTap: () => Navigator.pushNamed(
                            context, AppRoutes.equipmentRent),
                      ),
                      _PillChip(
                        icon: Icons.storefront_rounded,
                        label: "Market",
                        onTap: () =>
                            Navigator.pushNamed(context, AppRoutes.market),
                      ),
                      _PillChip(
                        icon: Icons.grass_rounded,
                        label: "Soil",
                        onTap: () =>
                            Navigator.pushNamed(context, AppRoutes.soilQuality),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 14),

                GridView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: actions.length,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                    childAspectRatio: 1.32,
                  ),
                  itemBuilder: (context, i) {
                    final a = actions[i];
                    return _ActionCard(
                      item: a,
                      onTap: () => Navigator.pushNamed(context, a.route!),
                    );
                  },
                ),
              ],
            ),
          ),

          const SizedBox(height: 18),

          // ===================== FEATURED STRIP (simple, modern) =====================
          // Container(
          //   width: double.infinity,
          //   margin: const EdgeInsets.symmetric(horizontal: 16),
          //   padding: const EdgeInsets.all(16),
          //   decoration: BoxDecoration(
          //     color: Colors.white,
          //     borderRadius: BorderRadius.circular(24),
          //     border: Border.all(color: AppColors.border),
          //   ),
          //   child: Row(
          //     children: [
          //       Container(
          //         width: 46,
          //         height: 46,
          //         decoration: BoxDecoration(
          //           color: AppColors.primary.withOpacity(0.18),
          //           borderRadius: BorderRadius.circular(16),
          //         ),
          //         child: const Icon(Icons.verified_rounded,
          //             color: AppColors.darkGreen),
          //       ),
          //       const SizedBox(width: 12),
          //       Expanded(
          //         child: Column(
          //           crossAxisAlignment: CrossAxisAlignment.start,
          //           children: [
          //             const Text(
          //               "Trusted network",
          //               style: TextStyle(
          //                 fontWeight: FontWeight.w900,
          //                 color: AppColors.textDark,
          //               ),
          //             ),
          //             const SizedBox(height: 2),
          //             Text(
          //               "Verified labour & equipment providers",
          //               style: TextStyle(
          //                 fontWeight: FontWeight.w700,
          //                 color: AppColors.textDark.withOpacity(0.60),
          //               ),
          //             ),
          //           ],
          //         ),
          //       ),
          //       const Icon(Icons.arrow_forward_ios_rounded,
          //           size: 16, color: AppColors.darkGreen),
          //     ],
          //   ),
          // ),

          const SizedBox(height: 22),
        ],
      ),
    );
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
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    fontSize: 14.6,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            "$season - $monthName Week $week",
            style: TextStyle(
              fontWeight: FontWeight.w700,
              color: AppColors.textDark.withOpacity(0.65),
              fontSize: 12.2,
            ),
          ),
          const SizedBox(height: 10),
          if (advisories.isEmpty)
            Text(
              "No HOLD/SELL alert for this week. Monitor market trend and check next week.",
              style: TextStyle(
                fontWeight: FontWeight.w600,
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
          backgroundColor: Colors.white.withOpacity(0.12),
          child: Icon(icon, color: Colors.white),
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
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    fontSize: 12.4,
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
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: Colors.white.withOpacity(0.92)),
          const SizedBox(height: 6),
          Text(
            value,
            style: TextStyle(
              color: Colors.white.withOpacity(0.96),
              fontWeight: FontWeight.w900,
              fontSize: 15.6,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            label,
            style: TextStyle(
              color: Colors.white.withOpacity(0.72),
              fontWeight: FontWeight.w700,
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
  final Color bg;
  final Color iconBg;
  final String? route;

  const _ActionItem({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.bg,
    required this.iconBg,
    this.route,
  });
}

class _ActionCard extends StatelessWidget {
  final _ActionItem item;
  final VoidCallback onTap;

  const _ActionCard({required this.item, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: item.bg,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
          child: Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: item.iconBg,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(item.icon, color: AppColors.darkGreen),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      item.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        color: AppColors.textDark,
                        fontSize: 13.4,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      item.subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: AppColors.textDark.withOpacity(0.55),
                        fontSize: 11.6,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.arrow_forward_ios_rounded,
                  size: 14, color: AppColors.textDark.withOpacity(0.35)),
            ],
          ),
        ),
      ),
    );
  }
}
