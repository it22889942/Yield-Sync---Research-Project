import 'package:flutter/material.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';
import '../../services/equipment_api.dart';

class EquipmentRentScreen extends StatefulWidget {
  const EquipmentRentScreen({super.key});

  @override
  State<EquipmentRentScreen> createState() => _EquipmentRentScreenState();
}

class _EquipmentRentScreenState extends State<EquipmentRentScreen> {
  final _searchCtrl = TextEditingController();

  // ✅ backend locations + types
  List<String> _locations = const ["Kurunegala"];
  List<String> _types = const ["All"];

  String _location = "Kurunegala";
  String _type = "All";

  bool _loadingLocs = true;
  bool _loadingTypes = true;

  // UX extras
  final List<String> _recent = [];
  final List<String> _popular = const [
    "Tractor",
    "Harvester",
    "Rotavator",
    "Sprayer",
    "Water Pump",
    "Plough",
    "Seeder",
  ];

  @override
  void initState() {
    super.initState();
    _loadLocations();
    _loadTypes();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  // ✅ "Tractor (4WD)" -> "Tractor"
  String _baseType(String full) {
    final i = full.indexOf("(");
    if (i == -1) return full.trim();
    return full.substring(0, i).trim();
  }

  Future<void> _loadLocations() async {
    setState(() => _loadingLocs = true);

    try {
      final locs = await EquipmentApi.getLocations();
      if (!mounted) return;

      setState(() {
        _locations = locs.isEmpty ? ["Kurunegala"] : locs;
        if (!_locations.contains(_location)) _location = _locations.first;
        _loadingLocs = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loadingLocs = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to load locations: $e")),
      );
    }
  }

  Future<void> _loadTypes() async {
    setState(() => _loadingTypes = true);

    try {
      final t = await EquipmentApi.getTypes();
      if (!mounted) return;

      final types = ["All", ...t.where((x) => x.trim().isNotEmpty)];
      setState(() {
        _types = types.isEmpty ? ["All"] : types;
        if (!_types.contains(_type)) _type = _types.first;
        _loadingTypes = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loadingTypes = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to load types: $e")),
      );
    }
  }

  void _goSearch() {
    final q = _searchCtrl.text.trim();

    if (q.isNotEmpty) {
      // store recent searches (simple local list)
      setState(() {
        _recent.removeWhere((x) => x.toLowerCase() == q.toLowerCase());
        _recent.insert(0, q);
        if (_recent.length > 6) _recent.removeLast();
      });
    }

    Navigator.pushNamed(
      context,
      AppRoutes.equipmentList,
      arguments: EquipmentSearchArgs(
        query: q,
        location: _location,
        // ✅ backend expects base type
        type: _type == "All" ? "" : _baseType(_type),
      ),
    );
  }

  void _openFiltersSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return _FilterSheet(
          loadingLocs: _loadingLocs,
          loadingTypes: _loadingTypes,
          locations: _locations,
          types: _types,
          initialLocation: _location,
          initialType: _type,
          onReloadLocs: _loadLocations,
          onReloadTypes: _loadTypes,
          onApply: (loc, type) {
            setState(() {
              _location = loc;
              _type = type;
            });
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===== MODERN HERO HEADER =====
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
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
                  top: 28,
                  child: Opacity(
                    opacity: 0.12,
                    child: Icon(Icons.agriculture_rounded,
                        size: 150, color: Colors.white),
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.arrow_back_ios_new_rounded),
                          color: Colors.white,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          "Rent Equipment",
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.92),
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                        const Spacer(),
                        CircleAvatar(
                          radius: 18,
                          backgroundColor: Colors.white.withOpacity(0.12),
                          child: const Icon(Icons.person_rounded,
                              color: Colors.white),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Find the Right Machine\nFast",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.96),
                        fontWeight: FontWeight.w900,
                        fontSize: 24,
                        height: 1.08,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      "Search by type, location and keywords.\nBook equipment from trusted owners.",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontWeight: FontWeight.w600,
                        fontSize: 12.8,
                        height: 1.22,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // Quick stats chips (your numbers)
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: const [
                        _HeroChip(
                            icon: Icons.handyman_rounded,
                            text: "500+ Equipment"),
                        _HeroChip(
                            icon: Icons.place_rounded, text: "25 Districts"),
                        _HeroChip(
                            icon: Icons.verified_rounded,
                            text: "Verified Owners"),
                      ],
                    ),

                    const SizedBox(height: 12),

                    // Image area (free space): banner card
                    ClipRRect(
                      borderRadius: BorderRadius.circular(18),
                      child: SizedBox(
                        height: 120,
                        width: double.infinity,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            // If you don't want assets, keep fallback icon
                            Image.asset(
                              "images/equipment_banner.jpg",
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => Container(
                                color: Colors.white.withOpacity(0.10),
                                child: const Center(
                                  child: Icon(Icons.local_shipping_rounded,
                                      color: AppColors.primary, size: 44),
                                ),
                              ),
                            ),
                            Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  begin: Alignment.topCenter,
                                  end: Alignment.bottomCenter,
                                  colors: [
                                    Colors.black.withOpacity(0.08),
                                    Colors.black.withOpacity(0.48),
                                  ],
                                ),
                              ),
                            ),
                            Positioned(
                              left: 12,
                              right: 12,
                              bottom: 12,
                              child: Row(
                                children: [
                                  _StatusPill(
                                    icon: Icons.location_on_rounded,
                                    text: "$_location",
                                  ),
                                  const SizedBox(width: 10),
                                  _StatusPill(
                                    icon: Icons.category_rounded,
                                    text: _type,
                                  ),
                                  const Spacer(),
                                  InkWell(
                                    borderRadius: BorderRadius.circular(999),
                                    onTap: _openFiltersSheet,
                                    child: CircleAvatar(
                                      radius: 18,
                                      backgroundColor:
                                          Colors.white.withOpacity(0.16),
                                      child: const Icon(Icons.tune_rounded,
                                          color: Colors.white),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // ===== BODY =====
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
              children: [
                // Search card
                Container(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(22),
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
                          const Expanded(
                            child: Text(
                              "Search Equipment",
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                fontSize: 16,
                                color: AppColors.textDark,
                              ),
                            ),
                          ),
                          TextButton.icon(
                            onPressed: _openFiltersSheet,
                            icon: const Icon(Icons.tune_rounded, size: 18),
                            label: const Text(
                              "Filters",
                              style: TextStyle(fontWeight: FontWeight.w900),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),

                      // ✅ Autocomplete using types
                      Autocomplete<String>(
                        optionsBuilder: (TextEditingValue text) {
                          final q = text.text.trim().toLowerCase();
                          if (q.isEmpty) return const Iterable<String>.empty();

                          return _types
                              .where((t) => t != "All")
                              .where((t) => t.toLowerCase().contains(q))
                              .take(8);
                        },
                        onSelected: (value) => _searchCtrl.text = value,
                        fieldViewBuilder:
                            (context, controller, focusNode, onSubmit) {
                          controller.text = _searchCtrl.text;
                          controller.addListener(() {
                            _searchCtrl.text = controller.text;
                          });

                          return TextField(
                            controller: controller,
                            focusNode: focusNode,
                            textInputAction: TextInputAction.search,
                            onSubmitted: (_) => _goSearch(),
                            decoration: InputDecoration(
                              hintText:
                                  "Search (tractor, 4wd, pump, sprayer...)",
                              prefixIcon: const Icon(Icons.search_rounded),
                              suffixIcon: IconButton(
                                onPressed: () {
                                  controller.clear();
                                  _searchCtrl.clear();
                                  setState(() {});
                                },
                                icon: const Icon(Icons.close_rounded),
                              ),
                              filled: true,
                              fillColor: AppColors.surface,
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide:
                                    const BorderSide(color: AppColors.border),
                              ),
                              enabledBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide:
                                    const BorderSide(color: AppColors.border),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide: const BorderSide(
                                  color: AppColors.primary,
                                  width: 1.6,
                                ),
                              ),
                            ),
                          );
                        },
                      ),

                      const SizedBox(height: 12),

                      // Compact active filters row (modern)
                      Row(
                        children: [
                          Expanded(
                            child: _FilterPill(
                              icon: Icons.location_on_rounded,
                              text: _loadingLocs ? "Loading..." : _location,
                              onTap: _openFiltersSheet,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: _FilterPill(
                              icon: Icons.category_rounded,
                              text: _loadingTypes ? "Loading..." : _type,
                              onTap: _openFiltersSheet,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 14),

                      // CTA
                      SizedBox(
                        width: double.infinity,
                        height: 52,
                        child: ElevatedButton.icon(
                          onPressed: _goSearch,
                          icon: const Icon(Icons.search_rounded),
                          label: const Text(
                            "Search Now",
                            style: TextStyle(fontWeight: FontWeight.w900),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primary,
                            foregroundColor: AppColors.darkGreen,
                            elevation: 0,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 12),

                // Recent searches
                if (_recent.isNotEmpty) ...[
                  const Text(
                    "Recent",
                    style: TextStyle(
                      fontWeight: FontWeight.w900,
                      color: AppColors.textDark,
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: _recent
                        .map(
                          (x) => _TagChip(
                            text: x,
                            onTap: () {
                              setState(() => _searchCtrl.text = x);
                              _goSearch();
                            },
                          ),
                        )
                        .toList(),
                  ),
                  const SizedBox(height: 14),
                ],

                // Popular categories
                const Text(
                  "Popular Categories",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 10),
                GridView.count(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  crossAxisCount: 2,
                  crossAxisSpacing: 12,
                  mainAxisSpacing: 12,
                  childAspectRatio: 2.7,
                  children: _popular.map((p) {
                    return _QuickCategoryCard(
                      text: p,
                      onTap: () {
                        setState(() => _searchCtrl.text = p);
                        _goSearch();
                      },
                    );
                  }).toList(),
                ),

                const SizedBox(height: 18),

                // Small help card
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: AppColors.border),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 42,
                        height: 42,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: const Icon(Icons.tips_and_updates_rounded,
                            color: AppColors.darkGreen),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          "Tip: Try keywords like “4WD”, “diesel”, “rice”, “mini tractor” to get better results.",
                          style: TextStyle(
                            color: AppColors.textDark.withOpacity(0.75),
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                    ],
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

// ===================== Bottom sheet filters =====================
class _FilterSheet extends StatefulWidget {
  final bool loadingLocs;
  final bool loadingTypes;
  final List<String> locations;
  final List<String> types;
  final String initialLocation;
  final String initialType;
  final VoidCallback onReloadLocs;
  final VoidCallback onReloadTypes;
  final void Function(String location, String type) onApply;

  const _FilterSheet({
    required this.loadingLocs,
    required this.loadingTypes,
    required this.locations,
    required this.types,
    required this.initialLocation,
    required this.initialType,
    required this.onReloadLocs,
    required this.onReloadTypes,
    required this.onApply,
  });

  @override
  State<_FilterSheet> createState() => _FilterSheetState();
}

class _FilterSheetState extends State<_FilterSheet> {
  late String _loc;
  late String _type;

  @override
  void initState() {
    super.initState();
    _loc = widget.initialLocation;
    _type = widget.initialType;
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(22),
          border: Border.all(color: AppColors.border),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.15),
              blurRadius: 28,
              offset: const Offset(0, 18),
            )
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // handle
            Container(
              width: 46,
              height: 5,
              decoration: BoxDecoration(
                color: AppColors.border,
                borderRadius: BorderRadius.circular(999),
              ),
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                const Expanded(
                  child: Text(
                    "Filters",
                    style: TextStyle(
                      fontWeight: FontWeight.w900,
                      fontSize: 16,
                      color: AppColors.textDark,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.close_rounded),
                ),
              ],
            ),
            const SizedBox(height: 8),

            // Location
            Row(
              children: [
                const Expanded(
                  child: Text(
                    "Location",
                    style: TextStyle(fontWeight: FontWeight.w900),
                  ),
                ),
                TextButton.icon(
                  onPressed: widget.onReloadLocs,
                  icon: const Icon(Icons.refresh_rounded, size: 18),
                  label: const Text(
                    "Reload",
                    style: TextStyle(fontWeight: FontWeight.w900),
                  ),
                )
              ],
            ),
            widget.loadingLocs
                ? _loadingBox()
                : DropdownButtonFormField<String>(
                    value: _loc,
                    items: widget.locations
                        .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                        .toList(),
                    onChanged: (v) => setState(() => _loc = v ?? _loc),
                    decoration: _ddDecoration(
                      label: "Select location",
                      icon: Icons.location_on_rounded,
                    ),
                  ),

            const SizedBox(height: 12),

            // Type
            Row(
              children: [
                const Expanded(
                  child: Text(
                    "Equipment Type",
                    style: TextStyle(fontWeight: FontWeight.w900),
                  ),
                ),
                TextButton.icon(
                  onPressed: widget.onReloadTypes,
                  icon: const Icon(Icons.refresh_rounded, size: 18),
                  label: const Text(
                    "Reload",
                    style: TextStyle(fontWeight: FontWeight.w900),
                  ),
                )
              ],
            ),
            widget.loadingTypes
                ? _loadingBox()
                : DropdownButtonFormField<String>(
                    value: _type,
                    items: widget.types
                        .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                        .toList(),
                    onChanged: (v) => setState(() => _type = v ?? _type),
                    decoration: _ddDecoration(
                      label: "Select type",
                      icon: Icons.category_rounded,
                    ),
                  ),

            const SizedBox(height: 14),

            SizedBox(
              width: double.infinity,
              height: 52,
              child: ElevatedButton(
                onPressed: () {
                  widget.onApply(_loc, _type);
                  Navigator.pop(context);
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: AppColors.darkGreen,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                child: const Text(
                  "Apply Filters",
                  style: TextStyle(fontWeight: FontWeight.w900),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _loadingBox() {
    return Container(
      height: 56,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: const SizedBox(
        width: 22,
        height: 22,
        child: CircularProgressIndicator(strokeWidth: 2.4),
      ),
    );
  }

  InputDecoration _ddDecoration(
      {required String label, required IconData icon}) {
    return InputDecoration(
      labelText: label,
      prefixIcon: Icon(icon),
      filled: true,
      fillColor: AppColors.surface,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: AppColors.primary, width: 1.6),
      ),
    );
  }
}

// ===================== Small UI parts =====================
class _HeroChip extends StatelessWidget {
  final IconData icon;
  final String text;
  const _HeroChip({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: AppColors.primary, size: 16),
          const SizedBox(width: 8),
          Text(
            text,
            style: TextStyle(
              color: Colors.white.withOpacity(0.90),
              fontWeight: FontWeight.w800,
              fontSize: 12.5,
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  final IconData icon;
  final String text;
  const _StatusPill({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.16),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: Colors.white.withOpacity(0.92), size: 16),
          const SizedBox(width: 8),
          Text(
            text,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w900,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

class _FilterPill extends StatelessWidget {
  final IconData icon;
  final String text;
  final VoidCallback onTap;

  const _FilterPill({
    required this.icon,
    required this.text,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppColors.surface,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          height: 50,
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              Icon(icon, color: AppColors.darkGreen, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  text,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                  ),
                ),
              ),
              Icon(Icons.keyboard_arrow_down_rounded,
                  color: AppColors.textDark.withOpacity(0.45)),
            ],
          ),
        ),
      ),
    );
  }
}

class _TagChip extends StatelessWidget {
  final String text;
  final VoidCallback onTap;
  const _TagChip({required this.text, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: AppColors.border),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: AppColors.textDark.withOpacity(0.85),
            fontWeight: FontWeight.w900,
            fontSize: 12.2,
          ),
        ),
      ),
    );
  }
}

class _QuickCategoryCard extends StatelessWidget {
  final String text;
  final VoidCallback onTap;

  const _QuickCategoryCard({required this.text, required this.onTap});

  IconData _icon(String t) {
    final x = t.toLowerCase();
    if (x.contains("tractor")) return Icons.agriculture_rounded;
    if (x.contains("harvest")) return Icons.grass_rounded;
    if (x.contains("spray")) return Icons.water_drop_rounded;
    if (x.contains("pump")) return Icons.water_rounded;
    if (x.contains("plough")) return Icons.handyman_rounded;
    if (x.contains("seed")) return Icons.spa_rounded;
    return Icons.category_rounded;
  }

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppColors.primary.withOpacity(0.10),
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              Container(
                width: 38,
                height: 38,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(_icon(text), color: AppColors.darkGreen, size: 20),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  text,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                  ),
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

// ===================== args (unchanged) =====================
class EquipmentSearchArgs {
  final String query;
  final String location;
  final String type; // "" => all

  EquipmentSearchArgs({
    required this.query,
    required this.location,
    required this.type,
  });
}
