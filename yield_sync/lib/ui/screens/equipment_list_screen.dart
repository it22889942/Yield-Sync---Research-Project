import 'dart:async';
import 'package:flutter/material.dart';

import '../../utils/app_colors.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';

import '../../services/equipment_api.dart';
import 'equipment_rent_screen.dart';

enum _SortMode { best, cheapest, rating }

class EquipmentListScreen extends StatefulWidget {
  const EquipmentListScreen({super.key});

  @override
  State<EquipmentListScreen> createState() => _EquipmentListScreenState();
}

class _EquipmentListScreenState extends State<EquipmentListScreen> {
  final _searchCtrl = TextEditingController();

  late EquipmentSearchArgs _args;

  bool _loading = true;
  String? _error;

  List<EquipmentListItem> _items = [];
  List<EquipmentListItem> _view = [];

  Timer? _debounce;

  _SortMode _sort = _SortMode.best;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();

    final a = ModalRoute.of(context)?.settings.arguments;
    _args = (a is EquipmentSearchArgs)
        ? a
        : EquipmentSearchArgs(query: "", location: "Kurunegala", type: "");

    _searchCtrl.text = _args.query;

    _fetch();
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final list = await EquipmentApi.search(
        query: _searchCtrl.text.trim(),
        location: _args.location,
        type: _args.type,
        topK: 30,
      );

      if (!mounted) return;

      setState(() {
        _items = list;
        _loading = false;
      });

      _applySortAndFilter();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  void _onSearchChanged(String _) {
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 450), _fetch);
  }

  void _applySortAndFilter() {
    final v = [..._items];

    switch (_sort) {
      case _SortMode.best:
        v.sort((a, b) {
          final sa = (a.rating * 2.2) + (a.pastBookings * 0.04);
          final sb = (b.rating * 2.2) + (b.pastBookings * 0.04);
          return sb.compareTo(sa);
        });
        break;
      case _SortMode.cheapest:
        v.sort((a, b) => a.dailyRate.compareTo(b.dailyRate));
        break;
      case _SortMode.rating:
        v.sort((a, b) => b.rating.compareTo(a.rating));
        break;
    }

    setState(() => _view = v);
  }

  String _subTitleText() {
    final typeLabel = _args.type.isEmpty ? "All types" : _args.type;
    final count = _loading ? "…" : "${_items.length}";
    return "${_args.location} • $typeLabel • $count items";
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===== HERO HEADER =====
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
            child: Column(
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
                      child:
                          const Icon(Icons.person_rounded, color: Colors.white),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Available Equipment",
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.96),
                      fontWeight: FontWeight.w900,
                      fontSize: 24,
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    _subTitleText(),
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.72),
                      fontWeight: FontWeight.w700,
                      fontSize: 12.8,
                    ),
                  ),
                ),
                const SizedBox(height: 12),

                // Header banner
                ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: SizedBox(
                    height: 110,
                    width: double.infinity,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.asset(
                          // ✅ use assets path (web-safe)
                          "assets/images/equipment_banner.jpg",
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
                                Colors.black.withOpacity(0.45),
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
                              _HeaderPill(
                                icon: Icons.location_on_rounded,
                                text: _args.location,
                              ),
                              const SizedBox(width: 10),
                              _HeaderPill(
                                icon: Icons.category_rounded,
                                text: _args.type.isEmpty ? "All" : _args.type,
                              ),
                              const Spacer(),
                              InkWell(
                                borderRadius: BorderRadius.circular(999),
                                onTap: _fetch,
                                child: CircleAvatar(
                                  radius: 18,
                                  backgroundColor:
                                      Colors.white.withOpacity(0.16),
                                  child: const Icon(Icons.refresh_rounded,
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
          ),

          const SizedBox(height: 12),

          // ===== SEARCH + SORT =====
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _searchCtrl,
                    onChanged: _onSearchChanged,
                    textInputAction: TextInputAction.search,
                    onSubmitted: (_) => _fetch(),
                    decoration: InputDecoration(
                      hintText: "Search equipment...",
                      prefixIcon: const Icon(Icons.search_rounded),
                      suffixIcon: IconButton(
                        onPressed: () {
                          _searchCtrl.clear();
                          _fetch();
                        },
                        icon: const Icon(Icons.close_rounded),
                      ),
                      filled: true,
                      fillColor: Colors.white,
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
                        borderSide: const BorderSide(
                            color: AppColors.primary, width: 1.6),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                _SortButton(
                  mode: _sort,
                  onChanged: (m) {
                    setState(() => _sort = m);
                    _applySortAndFilter();
                  },
                ),
              ],
            ),
          ),

          const SizedBox(height: 10),

          // ===== LIST =====
          Expanded(
            child: RefreshIndicator(
              onRefresh: _fetch,
              child: _loading
                  ? ListView.builder(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                      itemCount: 6,
                      itemBuilder: (_, __) => const _SkeletonCard(),
                    )
                  : _error != null
                      ? ListView(
                          padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
                          children: [
                            _ErrorBox(message: _error!, onRetry: _fetch),
                          ],
                        )
                      : _items.isEmpty
                          ? ListView(
                              padding:
                                  const EdgeInsets.fromLTRB(16, 24, 16, 16),
                              children: [
                                _EmptyBox(
                                  title: "No equipment found",
                                  subtitle:
                                      "Try another keyword or select another district/type.",
                                  onRetry: _fetch,
                                ),
                              ],
                            )
                          : ListView.builder(
                              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                              itemCount: _view.length,
                              itemBuilder: (context, i) {
                                final e = _view[i];
                                return _ModernEquipmentCard(
                                  item: e,
                                  onTap: () {
                                    Navigator.pushNamed(
                                      context,
                                      AppRoutes.equipmentDetails,
                                      arguments: e.id,
                                    );
                                  },
                                );
                              },
                            ),
            ),
          ),
        ],
      ),
    );
  }
}

// ===================== MODERN CARD =====================
class _ModernEquipmentCard extends StatelessWidget {
  final EquipmentListItem item;
  final VoidCallback onTap;

  const _ModernEquipmentCard({required this.item, required this.onTap});

  IconData _iconForType(String t) {
    final s = t.toLowerCase();
    if (s.contains("tractor")) return Icons.agriculture_rounded;
    if (s.contains("harvest")) return Icons.grass_rounded;
    if (s.contains("pump") || s.contains("water")) return Icons.water_rounded;
    if (s.contains("spray")) return Icons.water_drop_rounded;
    if (s.contains("plough")) return Icons.handyman_rounded;
    if (s.contains("seed")) return Icons.spa_rounded;
    if (s.contains("trailer")) return Icons.local_shipping_rounded;
    if (s.contains("transplant")) return Icons.eco_rounded;
    return Icons.build_rounded;
  }

  Color _badgeColor(String t) {
    final s = t.toLowerCase();
    if (s.contains("tractor")) return const Color(0xFF2FA36B);
    if (s.contains("harvest")) return const Color(0xFF7B61FF);
    if (s.contains("pump") || s.contains("water")) {
      return const Color(0xFF1C7ED6);
    }
    if (s.contains("spray")) return const Color(0xFFFF922B);
    return const Color(0xFF2D3748);
  }

  // ✅ normalize text for matching
  String _n(String x) => x.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), ' ');

  // ✅ choose local image from your assets/equpment folder (folder name from your screenshot)
  String? _assetForItem() {
    final full = _n(item.equipmentType);
    final crop = _n(item.forCrop);
    final all = "$full $crop";

    // Tractor
    if (all.contains("tractor")) {
      if (all.contains("new holland") || all.contains("3630")) {
        return "images/equpment/tractor-New-Holland3630TX.jpeg";
      }
      if (all.contains("mahindra") || all.contains("475")) {
        return "images/equpment/tractor-Mahindra475DI.jpg";
      }
      // default tractor image
      return "images/equpment/tractor-New-Holland3630TX.jpeg";
    }

    // Harvester / combine
    if (all.contains("harvest") || all.contains("combine")) {
      if (all.contains("kubota") || all.contains("dc 70")) {
        return "images/equpment/harvester-Kubota-DC-70.jpg";
      }
      if (all.contains("claas") || all.contains("tiger")) {
        return "images/equpment/harvester-CLAAS-CROP-TIGER.jpg";
      }
      return "images/equpment/combine-Harvester.webp";
    }

    // Rotavator
    if (all.contains("rotavator")) {
      if (all.contains("shaktiman")) {
        return "images/equpment/rotavator-Shaktiman-Side-Shift.png";
      }
      return "images/equpment/rotavator-Fieldking-FKRTMG.jpg";
    }

    // Plough
    if (all.contains("plough")) {
      if (all.contains("disc"))
        return "images/equpment/Plough-Disc3-Bottom.jpg";
      return "images/equpment/plough-Mould-Board2Bottom.jpg";
    }

    // Sprayer / power sprayer
    if (all.contains("spray")) {
      if (all.contains("neptune")) {
        return "images/equpment/sprayer-Neptune-HTP-Gold.webp";
      }
      if (all.contains("kisan")) {
        return "images/equpment/sprayer-KisanCraft16L.webp";
      }
      if (all.contains("power")) return "images/equpment/power-Sprayer.jpg";
      return "images/equpment/power-Sprayer.jpg";
    }

    // Seed drill
    if (all.contains("seed") || all.contains("drill")) {
      if (all.contains("mahindra")) {
        return "images/equpment/seed-Drill-Mahindra-PlantMaster.jpg";
      }
      return "images/equpment/seed-Drill-John-Deere-750A.jpg";
    }

    // Trailer
    if (all.contains("trailer")) {
      if (all.contains("single"))
        return "images/equpment/trailer-Single-Axle.jpg";
      return "images/equpment/trailer-Double-Axle.jpg";
    }

    // Transplanter
    if (all.contains("transplant")) {
      if (all.contains("yanmar"))
        return "images/equpment/transplanter-YANMAR-VP7.jpg";
      return "images/equpment/transplanter-Kubota-SPV-6MD.webp";
    }

    // Fertilizer spreader
    if (all.contains("fertilizer") || all.contains("spreader")) {
      return "images/equpment/fertilizer-Spreader (2).png";
    }

    // Grass cutter
    if (all.contains("grass") || all.contains("cutter")) {
      return "images/equpment/grass-Cutter.jpeg";
    }

    return null;
  }

  @override
  Widget build(BuildContext context) {
    final type = item.equipmentType.isEmpty ? "Equipment" : item.equipmentType;
    final loc = item.nearestMajorDistrict.isNotEmpty
        ? item.nearestMajorDistrict
        : item.location;
    final imageAsset = _assetForItem();

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 22,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      child: Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: onTap,
          child: Column(
            children: [
              // top image area
              Container(
                height: 150,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.10),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(20),
                    topRight: Radius.circular(20),
                  ),
                ),
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: ClipRRect(
                        borderRadius: const BorderRadius.only(
                          topLeft: Radius.circular(20),
                          topRight: Radius.circular(20),
                        ),
                        child: imageAsset != null
                            ? Image.asset(
                                imageAsset,
                                fit: BoxFit.cover,
                                errorBuilder: (_, __, ___) =>
                                    _fallbackTopVisual(type),
                              )
                            : _fallbackTopVisual(type),
                      ),
                    ),

                    // soft overlay for readability
                    Positioned.fill(
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: const BorderRadius.only(
                            topLeft: Radius.circular(20),
                            topRight: Radius.circular(20),
                          ),
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              Colors.black.withOpacity(0.03),
                              Colors.black.withOpacity(0.08),
                            ],
                          ),
                        ),
                      ),
                    ),

                    // type badge
                    Positioned(
                      left: 12,
                      top: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 7),
                        decoration: BoxDecoration(
                          color: Colors.white
                              .withOpacity(0.94), // ✅ solid readable bg
                          borderRadius: BorderRadius.circular(999),
                          border: Border.all(
                            color: _badgeColor(type).withOpacity(0.28),
                            width: 1.2,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.10),
                              blurRadius: 10,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.category_rounded,
                              size: 14,
                              color: _badgeColor(type), // ✅ strong icon color
                            ),
                            const SizedBox(width: 6),
                            ConstrainedBox(
                              constraints: const BoxConstraints(maxWidth: 170),
                              child: Text(
                                type,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: TextStyle(
                                  fontWeight: FontWeight.w900,
                                  fontSize: 11.8,
                                  color: _badgeColor(
                                      type), // ✅ readable text color
                                  shadows: [
                                    Shadow(
                                      color: Colors.white.withOpacity(0.35),
                                      blurRadius: 1,
                                    )
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                    // price badge
                    Positioned(
                      right: 12,
                      top: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 7),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.90),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              "LKR ${item.dailyRate.toStringAsFixed(0)}",
                              style: const TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.darkGreen,
                                fontSize: 13.5,
                              ),
                            ),
                            Text(
                              "per day",
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark.withOpacity(0.55),
                                fontSize: 11,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // content
              Padding(
                padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      type,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        fontSize: 16,
                        color: AppColors.textDark,
                      ),
                    ),
                    const SizedBox(height: 8),

                    // location + hourly
                    Row(
                      children: [
                        const Icon(Icons.location_on_rounded,
                            size: 16, color: AppColors.darkGreen),
                        const SizedBox(width: 4),
                        Expanded(
                          child: Text(
                            loc,
                            style: TextStyle(
                              fontWeight: FontWeight.w800,
                              color: AppColors.textDark.withOpacity(0.70),
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          "LKR ${item.hourlyRate.toStringAsFixed(0)}/hr",
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark.withOpacity(0.70),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 10),

                    Row(
                      children: [
                        ...List.generate(
                          5,
                          (i) => Icon(
                            i < item.rating.round()
                                ? Icons.star_rounded
                                : Icons.star_border_rounded,
                            size: 18,
                            color: const Color(0xFFF5B400),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          item.rating.toStringAsFixed(1),
                          style: const TextStyle(
                            fontWeight: FontWeight.w900,
                            color: AppColors.textDark,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Text(
                          "${item.pastBookings} bookings",
                          style: TextStyle(
                            color: AppColors.textDark.withOpacity(0.55),
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 10),

                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.person_rounded,
                              color: AppColors.darkGreen),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              item.ownerName,
                              style: const TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.textDark,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          Container(
                            width: 38,
                            height: 38,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(14),
                              border: Border.all(color: AppColors.border),
                            ),
                            child: Icon(Icons.call_rounded,
                                color: AppColors.darkGreen.withOpacity(0.9)),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 8),

                    Text(
                      "Contact: ${item.ownerContact} • ${item.condition}",
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: AppColors.textDark.withOpacity(0.55),
                      ),
                    ),

                    const SizedBox(height: 12),

                    SizedBox(
                      width: double.infinity,
                      height: 46,
                      child: ElevatedButton(
                        onPressed: onTap,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: AppColors.darkGreen,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                        child: const Text(
                          "View Details",
                          style: TextStyle(fontWeight: FontWeight.w900),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _fallbackTopVisual(String type) {
    return Container(
      color: AppColors.primary.withOpacity(0.10),
      child: Center(
        child: Icon(
          _iconForType(type),
          size: 72,
          color: AppColors.darkGreen.withOpacity(0.86),
        ),
      ),
    );
  }
}

// ===================== SORT BUTTON =====================
class _SortButton extends StatelessWidget {
  final _SortMode mode;
  final ValueChanged<_SortMode> onChanged;

  const _SortButton({required this.mode, required this.onChanged});

  String _label(_SortMode m) {
    switch (m) {
      case _SortMode.best:
        return "Best";
      case _SortMode.cheapest:
        return "Cheap";
      case _SortMode.rating:
        return "Rating";
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<_SortMode>(
      onSelected: onChanged,
      itemBuilder: (_) => [
        PopupMenuItem(
            value: _SortMode.best, child: Text(_label(_SortMode.best))),
        PopupMenuItem(
            value: _SortMode.cheapest, child: Text(_label(_SortMode.cheapest))),
        PopupMenuItem(
            value: _SortMode.rating, child: Text(_label(_SortMode.rating))),
      ],
      child: Container(
        height: 56,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          children: [
            const Icon(Icons.sort_rounded),
            const SizedBox(width: 8),
            Text(
              _label(mode),
              style: const TextStyle(fontWeight: FontWeight.w900),
            ),
            const SizedBox(width: 6),
            Icon(Icons.keyboard_arrow_down_rounded,
                color: AppColors.textDark.withOpacity(0.55)),
          ],
        ),
      ),
    );
  }
}

// ===================== HEADER PILLS =====================
class _HeaderPill extends StatelessWidget {
  final IconData icon;
  final String text;
  const _HeaderPill({required this.icon, required this.text});

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

// ===================== SKELETON LOADER =====================
class _SkeletonCard extends StatelessWidget {
  const _SkeletonCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: [
          Container(
            height: 150,
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(20),
                topRight: Radius.circular(20),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              children: [
                _skLine(w: double.infinity, h: 14),
                const SizedBox(height: 10),
                _skLine(w: 220, h: 12),
                const SizedBox(height: 14),
                _skLine(w: double.infinity, h: 44),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _skLine({required double w, required double h}) {
    return Container(
      width: w,
      height: h,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(10),
      ),
    );
  }
}

// ===================== EMPTY/ERROR =====================
class _EmptyBox extends StatelessWidget {
  final String title;
  final String subtitle;
  final VoidCallback onRetry;

  const _EmptyBox({
    required this.title,
    required this.subtitle,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: const TextStyle(
                fontWeight: FontWeight.w900,
                fontSize: 16,
                color: AppColors.textDark,
              )),
          const SizedBox(height: 6),
          Text(
            subtitle,
            style: TextStyle(
              fontWeight: FontWeight.w700,
              color: AppColors.textDark.withOpacity(0.60),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text(
                "Retry",
                style: TextStyle(fontWeight: FontWeight.w900),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: AppColors.darkGreen,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}

class _ErrorBox extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorBox({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Error loading equipment",
            style: TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 16,
              color: AppColors.textDark,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            message,
            style: TextStyle(
              fontWeight: FontWeight.w700,
              color: AppColors.textDark.withOpacity(0.60),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text(
                "Retry",
                style: TextStyle(fontWeight: FontWeight.w900),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: AppColors.darkGreen,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}
