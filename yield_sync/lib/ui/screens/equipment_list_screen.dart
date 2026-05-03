import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../utils/app_colors.dart';
import '../../utils/equipment_image_asset.dart';
import '../../services/app_routes.dart';
import '../widgets/app_shell.dart';

import '../../services/equipment_api.dart';
import '../../services/recommendation_api.dart';
import 'equipment_rent_screen.dart';

enum _SortMode { best, cheapest, rating }

class EquipmentListScreen extends StatefulWidget {
  const EquipmentListScreen({super.key});

  @override
  State<EquipmentListScreen> createState() => _EquipmentListScreenState();
}

class _EquipmentListScreenState extends State<EquipmentListScreen> {
  final _searchCtrl = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  late EquipmentSearchArgs _args;

  bool _showScrollToTop = false;

  static const double _scrollToTopThreshold = 160;

  List<String> _locations = const ["Kurunegala"];
  List<String> _types = const [];
  String _selectedLocation = "Kurunegala";
  String? _selectedType;
  bool _loadingMeta = false;
  bool _metaInit = false;

  /// Only apply route arguments once (same pattern as labour list).
  bool _routeArgsApplied = false;

  static const double _equipmentWrapSpacing = 12;

  /// Two columns: each row uses [IntrinsicHeight] so both tiles match the taller card.
  Widget _equipmentCardsWrap(List<EquipmentListItem> items) {
    if (items.isEmpty) return const SizedBox.shrink();

    final rows = <Widget>[];
    for (var i = 0; i < items.length; i += 2) {
      if (i > 0) rows.add(SizedBox(height: _equipmentWrapSpacing));

      final left = items[i];
      final right = i + 1 < items.length ? items[i + 1] : null;

      rows.add(
        IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: _ModernEquipmentCard(
                  item: left,
                  onTap: () {
                    Navigator.pushNamed(
                      context,
                      AppRoutes.equipmentDetails,
                      arguments: left.id,
                    );
                  },
                ),
              ),
              SizedBox(width: _equipmentWrapSpacing),
              Expanded(
                child: right != null
                    ? _ModernEquipmentCard(
                        item: right,
                        onTap: () {
                          Navigator.pushNamed(
                            context,
                            AppRoutes.equipmentDetails,
                            arguments: right.id,
                          );
                        },
                      )
                    : const SizedBox.shrink(),
              ),
            ],
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: rows,
    );
  }

  Widget _equipmentSkeletonGrid() {
    return Column(
      children: [
        for (var r = 0; r < 3; r++) ...[
          if (r > 0) SizedBox(height: _equipmentWrapSpacing),
          IntrinsicHeight(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Expanded(child: _SkeletonCard()),
                SizedBox(width: _equipmentWrapSpacing),
                const Expanded(child: _SkeletonCard()),
              ],
            ),
          ),
        ],
      ],
    );
  }

  bool _loading = false;
  String? _error;

  List<EquipmentListItem> _items = [];
  List<EquipmentListItem> _view = [];

  /// When true: recommendation + semantic. When false: normal keyword search.
  bool _useSemanticSearch = true;

  _SortMode _sort = _SortMode.best;
  String? _primaryLocationFromQuery;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_routeArgsApplied) {
      _routeArgsApplied = true;
      final a = ModalRoute.of(context)?.settings.arguments;
      _args = (a is EquipmentSearchArgs)
          ? a
          : EquipmentSearchArgs(query: "", location: "Kurunegala", type: "");

      _searchCtrl.text = _args.query;
      _selectedLocation = _args.location;
      final typeArg = _args.type.trim();
      _selectedType = typeArg.isEmpty ? null : typeArg;

      if (!_metaInit) {
        _metaInit = true;
        _loadMeta();
      }
      _fetch();
    }
  }

  Future<void> _loadMeta() async {
    setState(() => _loadingMeta = true);
    try {
      final locs = await EquipmentApi.getLocations();
      final types = await EquipmentApi.getTypes();
      if (!mounted) return;
      setState(() {
        _locations = locs.isEmpty ? const ["Kurunegala"] : locs;
        _types = types
            .where(
              (s) =>
                  s.trim().isNotEmpty &&
                  s.trim().toLowerCase() != "unknown",
            )
            .toList();
        if (!_locations.contains(_selectedLocation)) {
          _selectedLocation = _locations.first;
        }
      });
    } catch (_) {
      // keep fallbacks
    } finally {
      if (mounted) setState(() => _loadingMeta = false);
    }
  }

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onEquipmentListScroll);
  }

  void _onEquipmentListScroll() {
    if (!_scrollController.hasClients) return;
    final show = _scrollController.offset > _scrollToTopThreshold;
    if (show != _showScrollToTop && mounted) {
      setState(() => _showScrollToTop = show);
    }
  }

  @override
  void dispose() {
    _scrollController.removeListener(_onEquipmentListScroll);
    _scrollController.dispose();
    _searchCtrl.dispose();
    super.dispose();
  }

  void _commitAndSearch() => _fetch();

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
      _args = EquipmentSearchArgs(
        query: _searchCtrl.text.trim(),
        location: _selectedLocation,
        type: _selectedType ?? "",
      );
    });

    try {
      final q = _searchCtrl.text.trim();
      _primaryLocationFromQuery = _extractLocationFromQuery(q);

      List<EquipmentListItem> list;

      if (_useSemanticSearch) {
        final parts = <String>[];
        if (q.isNotEmpty) parts.add(q);
        if (_args.type.trim().isNotEmpty) {
          parts.add(_args.type.trim());
        }
        if (_args.location.trim().isNotEmpty) {
          parts.add('in ${_args.location.trim()}');
        }
        final composedQuery = parts.join(' ').trim();

        final recs = await RecommendationApi.recommendEquipment(
          query: composedQuery.isEmpty ? _args.location : composedQuery,
          topK: 30,
        );
        list = recs
            .map((e) =>
                EquipmentListItem.fromJson(Map<String, dynamic>.from(e)))
            .toList();
      } else {
        list = await EquipmentApi.search(
          query: q,
          location: _args.location,
          type: _args.type,
          topK: 30,
        );
      }

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

  String? _extractLocationFromQuery(String raw) {
    final q = raw.trim();
    if (q.isEmpty) return null;
    final exp = RegExp(r'\bin\s+([A-Za-z\s]+)$', caseSensitive: false);
    final m = exp.firstMatch(q);
    if (m == null) return null;
    final loc = m.group(1)?.trim();
    if (loc == null || loc.isEmpty) return null;
    return loc;
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

  // ================= HERO (aligned with labour list screen) =================
  Widget _heroHeader() {
    final typeLabel =
        (_selectedType == null || _selectedType!.trim().isEmpty)
            ? "Any type"
            : _selectedType!.trim();

    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    _roundIconBtn(
                      icon: Icons.arrow_back_ios_new_rounded,
                      onTap: () => Navigator.pop(context),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        "Rent Equipment",
                        style: GoogleFonts.poppins(
                          color: Colors.white.withOpacity(0.95),
                          fontWeight: FontWeight.w700,
                          fontSize: 16,
                          letterSpacing: 0.15,
                        ),
                      ),
                    ),
                    _roundIconBtn(
                      icon: Icons.refresh_rounded,
                      onTap: () {
                        _loadMeta();
                        _fetch();
                      },
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Find Equipment",
                    style: GoogleFonts.poppins(
                      color: Colors.white.withOpacity(0.97),
                      fontWeight: FontWeight.w600,
                      fontSize: 21,
                      height: 1.12,
                      letterSpacing: -0.2,
                    ),
                  ),
                ),
                const SizedBox(height: 2),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "$typeLabel • $_selectedLocation",
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.inter(
                      color: Colors.white.withOpacity(0.82),
                      fontWeight: FontWeight.w500,
                      fontSize: 12.5,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Image.asset(
            'assets/images/equipmenthead.png',
            width: double.infinity,
            fit: BoxFit.fitWidth,
            alignment: Alignment.topCenter,
            gaplessPlayback: true,
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
            child: Container(
              padding: const EdgeInsets.fromLTRB(11, 9, 11, 10),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 12,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Theme(
                data: Theme.of(context).copyWith(
                  visualDensity: VisualDensity.compact,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          "Search & filter",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 13.5,
                            color: AppColors.textDark,
                          ),
                        ),
                        const Spacer(),
                        Text(
                          _useSemanticSearch ? "Smart" : "Keyword",
                          style: GoogleFonts.inter(
                            fontSize: 11.5,
                            color: AppColors.textDark.withOpacity(0.72),
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(width: 2),
                        Transform.scale(
                          scale: 0.82,
                          alignment: Alignment.center,
                          child: Switch(
                            value: _useSemanticSearch,
                            onChanged: (v) {
                              setState(() => _useSemanticSearch = v);
                              _fetch();
                            },
                            activeColor: AppColors.primary,
                            materialTapTargetSize:
                                MaterialTapTargetSize.shrinkWrap,
                          ),
                        ),
                        if (_loadingMeta || _loading) ...[
                          const SizedBox(width: 4),
                          const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 6),
                    _equipmentListSearchField(),
                    const SizedBox(height: 14),
                    _equipmentListLocationField(),
                    const SizedBox(height: 14),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          "Type",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 11.5,
                            color: AppColors.textDark.withOpacity(0.72),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: SizedBox(
                            height: 32,
                            child: _equipmentListTypeChipsHorizontal(),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          "Sort by",
                          style: GoogleFonts.poppins(
                            fontWeight: FontWeight.w700,
                            fontSize: 11.5,
                            color: AppColors.textDark.withOpacity(0.72),
                          ),
                        ),
                        const Spacer(),
                        _SortButton(
                          mode: _sort,
                          onChanged: (m) {
                            setState(() => _sort = m);
                            _applySortAndFilter();
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _equipmentListSearchField() {
    return TextField(
      controller: _searchCtrl,
      textInputAction: TextInputAction.search,
      style: GoogleFonts.inter(
        fontSize: 14,
        height: 1.3,
        fontWeight: FontWeight.w500,
        color: AppColors.textDark,
      ),
      onSubmitted: (_) => _commitAndSearch(),
      decoration: InputDecoration(
        isDense: true,
        hintText: _useSemanticSearch
            ? "e.g. harvester below 5000…"
            : "Name or equipment type…",
        hintStyle: GoogleFonts.inter(
          fontSize: 13,
          color: AppColors.textDark.withOpacity(0.45),
          fontWeight: FontWeight.w500,
        ),
        prefixIcon: const Icon(Icons.search_rounded, size: 20),
        prefixIconConstraints:
            const BoxConstraints(minWidth: 40, minHeight: 36),
        suffixIcon: _searchCtrl.text.trim().isEmpty
            ? null
            : IconButton(
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(
                  minWidth: 36,
                  minHeight: 36,
                ),
                onPressed: () {
                  setState(() => _searchCtrl.clear());
                  _commitAndSearch();
                },
                icon: const Icon(Icons.close_rounded, size: 20),
              ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide:
              const BorderSide(color: AppColors.primary, width: 1.4),
        ),
      ),
      onChanged: (_) => setState(() {}),
    );
  }

  Widget _equipmentListLocationField() {
    return DropdownButtonFormField<String>(
      isDense: true,
      value: _locations.contains(_selectedLocation)
          ? _selectedLocation
          : _locations.first,
      items: _locations
          .map(
            (e) => DropdownMenuItem(
              value: e,
              child: Text(
                e,
                style: GoogleFonts.inter(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          )
          .toList(),
      onChanged: (v) {
        if (v == null) return;
        setState(() => _selectedLocation = v);
        _commitAndSearch();
      },
      style: GoogleFonts.inter(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        color: AppColors.textDark,
      ),
      decoration: InputDecoration(
        labelText: "Your Location",
        labelStyle: GoogleFonts.inter(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
        ),
        prefixIcon: const Icon(Icons.location_on_rounded, size: 20),
        prefixIconConstraints:
            const BoxConstraints(minWidth: 40, minHeight: 36),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        filled: true,
        fillColor: AppColors.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide:
              const BorderSide(color: AppColors.primary, width: 1.4),
        ),
      ),
    );
  }

  Widget _equipmentListTypeChipsHorizontal() {
    final allTypes = _types
        .where((s) =>
            s.trim().isNotEmpty &&
            s.trim().toLowerCase() != "unknown")
        .toList();

    if (_loadingMeta && allTypes.isEmpty) {
      return ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: 6,
        separatorBuilder: (_, __) => const SizedBox(width: 6),
        itemBuilder: (_, __) => Container(
          width: 72,
          height: 28,
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: AppColors.border),
          ),
        ),
      );
    }

    return ListView(
      scrollDirection: Axis.horizontal,
      children: [
        _typePillChip(
          label: "Any type",
          selected: _selectedType == null,
          onTap: () {
            setState(() => _selectedType = null);
            _commitAndSearch();
          },
          icon: Icons.all_inclusive_rounded,
        ),
        ...allTypes.take(36).map((s) {
          final selected = _selectedType == s;
          return Padding(
            padding: const EdgeInsets.only(left: 6),
            child: _typePillChip(
              label: s,
              selected: selected,
              onTap: () {
                setState(() => _selectedType = selected ? null : s);
                _commitAndSearch();
              },
              icon: Icons.precision_manufacturing_rounded,
            ),
          );
        }),
      ],
    );
  }

  Widget _typePillChip({
    required String label,
    required bool selected,
    required VoidCallback onTap,
    required IconData icon,
  }) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
        decoration: BoxDecoration(
          color: selected
              ? AppColors.primary.withOpacity(0.18)
              : AppColors.surface,
          borderRadius: BorderRadius.circular(999),
          border: Border.all(
            color: selected ? AppColors.primary : AppColors.border,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 13,
              color: selected
                  ? AppColors.darkGreen
                  : AppColors.textDark.withOpacity(0.55),
            ),
            const SizedBox(width: 4),
            ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 120),
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.inter(
                  fontWeight: FontWeight.w600,
                  fontSize: 11.5,
                  height: 1.15,
                  color: selected
                      ? AppColors.textDark
                      : AppColors.textDark.withOpacity(0.75),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _roundIconBtn({required IconData icon, required VoidCallback onTap}) {
    return InkWell(
      borderRadius: BorderRadius.circular(999),
      onTap: onTap,
      child: CircleAvatar(
        radius: 18,
        backgroundColor: Colors.white.withOpacity(0.14),
        child: Icon(icon, color: Colors.white),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final primaryKey = _primaryLocationFromQuery?.toLowerCase().trim();
    final list = _view;

    List<EquipmentListItem> primary = list;
    List<EquipmentListItem> others = const [];

    if (primaryKey != null && primaryKey.isNotEmpty) {
      bool matches(EquipmentListItem e) {
        final loc = (e.nearestMajorDistrict.isNotEmpty
                ? e.nearestMajorDistrict
                : e.location)
            .toLowerCase();
        return loc.contains(primaryKey);
      }

      primary = list.where(matches).toList();
      others = list.where((e) => !matches(e)).toList();
    }

    return AppShell(
      currentIndex: 0,
      child: SizedBox.expand(
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned.fill(
              child: RefreshIndicator(
                onRefresh: _fetch,
                child: CustomScrollView(
                  controller: _scrollController,
                  physics: const AlwaysScrollableScrollPhysics(
                    parent: BouncingScrollPhysics(),
                  ),
                  slivers: [
                    SliverToBoxAdapter(child: _heroHeader()),
                    ..._equipmentListResultSlivers(primary, others),
                  ],
                ),
              ),
            ),
            if (_showScrollToTop)
              Positioned(
                right: 14,
                bottom: 14,
                child: Material(
                  elevation: 6,
                  shadowColor: Colors.black26,
                  shape: const CircleBorder(),
                  color: AppColors.primary,
                  clipBehavior: Clip.antiAlias,
                  child: InkWell(
                    customBorder: const CircleBorder(),
                    onTap: () {
                      _scrollController.animateTo(
                        0,
                        duration: const Duration(milliseconds: 420),
                        curve: Curves.easeOutCubic,
                      );
                    },
                    child: const Padding(
                      padding: EdgeInsets.all(11),
                      child: Icon(
                        Icons.keyboard_arrow_up_rounded,
                        color: AppColors.darkGreen,
                        size: 26,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  List<Widget> _equipmentListResultSlivers(
    List<EquipmentListItem> primary,
    List<EquipmentListItem> others,
  ) {
    if (_loading) {
      return [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          sliver: SliverToBoxAdapter(
            child: _equipmentSkeletonGrid(),
          ),
        ),
      ];
    }
    if (_error != null) {
      return [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          sliver: SliverToBoxAdapter(
            child: _ErrorBox(message: _error!, onRetry: _fetch),
          ),
        ),
      ];
    }
    if (_items.isEmpty) {
      return [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          sliver: SliverToBoxAdapter(
            child: _EmptyBox(
              title: "No equipment found",
              subtitle:
                  "Try another keyword or select another district/type.",
              onRetry: _fetch,
            ),
          ),
        ),
      ];
    }

    final slivers = <Widget>[];

    if (primary.isNotEmpty) {
      slivers.add(
        SliverPadding(
          padding: EdgeInsets.fromLTRB(
            16,
            12,
            16,
            others.isNotEmpty ? 0 : 16,
          ),
          sliver: SliverToBoxAdapter(
            child: _equipmentCardsWrap(primary),
          ),
        ),
      );
    }

    if (primary.isNotEmpty && others.isNotEmpty) {
      slivers.add(
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 18, 16, 10),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(0.08),
                borderRadius: BorderRadius.circular(999),
                border: Border.all(
                  color: AppColors.primary.withOpacity(0.45),
                  width: 1.2,
                ),
              ),
              child: Row(
                children: [
                  const Expanded(
                    child: Divider(
                      color: AppColors.border,
                      thickness: 1,
                    ),
                  ),
                  const SizedBox(width: 10),
                  const Icon(
                    Icons.location_city_rounded,
                    size: 16,
                    color: AppColors.darkGreen,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    "Other locations",
                    style: GoogleFonts.poppins(
                      fontWeight: FontWeight.w700,
                      color: AppColors.textDark,
                      fontSize: 12.5,
                    ),
                  ),
                  const SizedBox(width: 10),
                  const Expanded(
                    child: Divider(
                      color: AppColors.border,
                      thickness: 1,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    if (others.isNotEmpty) {
      slivers.add(
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          sliver: SliverToBoxAdapter(
            child: _equipmentCardsWrap(others),
          ),
        ),
      );
    }

    return slivers;
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

  @override
  Widget build(BuildContext context) {
    final type = item.equipmentType.isEmpty ? "Equipment" : item.equipmentType;
    final loc = item.nearestMajorDistrict.isNotEmpty
        ? item.nearestMajorDistrict
        : item.location;
    final imageAsset = EquipmentImageAsset.resolve(
      equipmentType: item.equipmentType,
      forCrop: item.forCrop,
      id: item.id,
    );

    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Material(
        color: Colors.white,
        clipBehavior: Clip.antiAlias,
        borderRadius: BorderRadius.circular(18),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Photo strip: use contain so whole equipment stays visible (cover was cropping).
              Container(
                height: 142,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.10),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(18),
                    topRight: Radius.circular(18),
                  ),
                ),
                child: ClipRRect(
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(18),
                    topRight: Radius.circular(18),
                  ),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Positioned.fill(
                        child: Image.asset(
                          imageAsset,
                          fit: BoxFit.contain,
                          alignment: Alignment.center,
                          filterQuality: FilterQuality.medium,
                          errorBuilder: (_, __, ___) =>
                              _fallbackTopVisual(type),
                        ),
                      ),
                      // Bottom scrim so chips stay readable without covering the whole photo.
                      Positioned(
                        left: 0,
                        right: 0,
                        bottom: 0,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [
                                Colors.transparent,
                                Colors.black.withOpacity(0.06),
                                Colors.black.withOpacity(0.24),
                              ],
                            ),
                          ),
                          child: Padding(
                            padding:
                                const EdgeInsets.fromLTRB(6, 14, 6, 6),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.end,
                              children: [
                                Flexible(
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 8, vertical: 5),
                                    decoration: BoxDecoration(
                                      color: Colors.white.withOpacity(0.96),
                                      borderRadius: BorderRadius.circular(999),
                                      border: Border.all(
                                        color: _badgeColor(type)
                                            .withOpacity(0.28),
                                        width: 1,
                                      ),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.black.withOpacity(0.08),
                                          blurRadius: 6,
                                          offset: const Offset(0, 2),
                                        ),
                                      ],
                                    ),
                                    child: Row(
                                      children: [
                                        Icon(
                                          Icons.category_rounded,
                                          size: 12,
                                          color: _badgeColor(type),
                                        ),
                                        const SizedBox(width: 4),
                                        Expanded(
                                          child: Text(
                                            type,
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                            style: GoogleFonts.poppins(
                                              fontWeight: FontWeight.w700,
                                              fontSize: 10,
                                              height: 1.12,
                                              color: _badgeColor(type),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 5),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.96),
                                    borderRadius: BorderRadius.circular(12),
                                    border:
                                        Border.all(color: AppColors.border),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.black.withOpacity(0.08),
                                        blurRadius: 6,
                                        offset: const Offset(0, 2),
                                      ),
                                    ],
                                  ),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.end,
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Text(
                                        "LKR ${item.dailyRate.toStringAsFixed(0)}",
                                        style: GoogleFonts.poppins(
                                          fontWeight: FontWeight.w700,
                                          color: AppColors.darkGreen,
                                          fontSize: 11,
                                        ),
                                      ),
                                      Text(
                                        "day",
                                        style: GoogleFonts.inter(
                                          fontWeight: FontWeight.w600,
                                          color: AppColors.textDark
                                              .withOpacity(0.55),
                                          fontSize: 9,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // Content fills remaining height so paired grid tiles align.
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      Text(
                        type,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.poppins(
                          fontWeight: FontWeight.w700,
                          fontSize: 13.5,
                          height: 1.12,
                          color: AppColors.textDark,
                          letterSpacing: -0.15,
                        ),
                      ),
                      const SizedBox(height: 5),

                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Padding(
                            padding: EdgeInsets.only(top: 2),
                            child: Icon(Icons.location_on_rounded,
                                size: 14, color: AppColors.darkGreen),
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            child: Text(
                              loc,
                              style: GoogleFonts.inter(
                                fontWeight: FontWeight.w500,
                                fontSize: 12.8,
                                color: AppColors.textDark.withOpacity(0.68),
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 3),

                      Text(
                        "LKR ${item.hourlyRate.toStringAsFixed(0)}/hr",
                        style: GoogleFonts.inter(
                          fontWeight: FontWeight.w600,
                          fontSize: 11,
                          color: AppColors.textDark.withOpacity(0.78),
                        ),
                      ),

                      const SizedBox(height: 6),

                      Row(
                        children: [
                          ...List.generate(
                            5,
                            (i) => Icon(
                              i < item.rating.round()
                                  ? Icons.star_rounded
                                  : Icons.star_border_rounded,
                              size: 13,
                              color: const Color(0xFFF5B400),
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            item.rating.toStringAsFixed(1),
                            style: GoogleFonts.inter(
                              fontWeight: FontWeight.w600,
                              fontSize: 12,
                              color: AppColors.textDark,
                            ),
                          ),
                        ],
                      ),

                      if (item.condition.trim().isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          item.condition,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.inter(
                            fontWeight: FontWeight.w500,
                            fontSize: 10,
                            height: 1.15,
                            color: AppColors.textDark.withOpacity(0.55),
                          ),
                        ),
                      ],
                    ],
                  ),
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
          size: 52,
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
        height: 40,
        padding: const EdgeInsets.symmetric(horizontal: 10),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.sort_rounded, size: 20),
            const SizedBox(width: 6),
            Text(
              _label(mode),
              style: GoogleFonts.poppins(
                fontWeight: FontWeight.w700,
                fontSize: 13,
                color: AppColors.textDark,
              ),
            ),
            const SizedBox(width: 4),
            Icon(
              Icons.keyboard_arrow_down_rounded,
              size: 20,
              color: AppColors.textDark.withOpacity(0.55),
            ),
          ],
        ),
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
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            height: 142,
            decoration: const BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(18),
                topRight: Radius.circular(18),
              ),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(10, 10, 10, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _skLine(w: double.infinity, h: 12),
                  const SizedBox(height: 8),
                  _skLine(w: double.infinity, h: 10),
                  const SizedBox(height: 8),
                  _skLine(w: 80, h: 10),
                  const SizedBox(height: 8),
                  _skLine(w: double.infinity, h: 12),
                ],
              ),
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
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.poppins(
              fontWeight: FontWeight.w700,
              fontSize: 17,
              color: AppColors.textDark,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            subtitle,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 13.2,
              height: 1.4,
              color: AppColors.textDark.withOpacity(0.65),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: Text(
                "Retry",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.15,
                ),
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
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.06),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.red.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Error loading equipment",
            style: GoogleFonts.poppins(
              fontWeight: FontWeight.w700,
              fontSize: 17,
              color: Colors.red,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            message,
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w500,
              fontSize: 13.2,
              height: 1.35,
              color: Colors.red.withOpacity(0.88),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 46,
            child: ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh_rounded),
              label: Text(
                "Retry",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.15,
                ),
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
