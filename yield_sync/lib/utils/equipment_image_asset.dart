/// Maps free-text equipment type / crop labels to bundled photos under
/// `assets/images/equipment/`.
class EquipmentImageAsset {
  EquipmentImageAsset._();

  static const String _dir = 'assets/images/equipment';

  static String _normalize(String raw) {
    return raw
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), ' ')
        .trim();
  }

  static bool _has(String normalized, List<String> keys) {
    return keys.any((k) => normalized.contains(k));
  }

  /// Always returns a path under [_dir]; falls back to a generic tractor photo.
  static String resolve({
    required String equipmentType,
    required String forCrop,
    String id = '',
  }) {
    final n = _normalize('$equipmentType $forCrop $id');

    if (_has(n, ['transplant', 'transplanter', 'rice plant'])) {
      return '$_dir/transplanter-YANMAR-VP7.jpg';
    }
    if (_has(n, ['trailer'])) {
      return '$_dir/trailer-Single-Axle.jpg';
    }
    if (_has(n, ['rotavat', 'rotavator', 'rotary tiller'])) {
      return '$_dir/rotavator-Fieldking-FKRTMG.jpg';
    }
    if (_has(n, ['fertilizer', 'fertiliser', 'spreader'])) {
      return '$_dir/fertilizer-Spreader (2).png';
    }
    if (_has(n, ['grass', 'cutter', 'brush cut'])) {
      return '$_dir/grass-Cutter.jpeg';
    }
    if (_has(n, ['disc']) && _has(n, ['plough', 'plow'])) {
      return '$_dir/Plough-Disc3-Bottom.jpg';
    }
    if (_has(n, ['plough', 'plow', 'mould', 'mold board'])) {
      return '$_dir/plough-Mould-Board2Bottom.jpg';
    }
    if (_has(n, ['seed', 'drill']) && !_has(n, ['spray'])) {
      return '$_dir/seed-Drill-John-Deere-750A.jpg';
    }
    if (_has(n, ['sprayer', 'spray', 'htp', 'neptune', 'kisan'])) {
      return '$_dir/power-Sprayer.jpg';
    }
    if (_has(n, ['pump', 'irrigation', 'water'])) {
      return '$_dir/power-Sprayer.jpg';
    }
    if (_has(n, ['combine', 'claas', 'crop tiger'])) {
      return '$_dir/combine-Harvester.webp';
    }
    if (_has(n, ['harvest', 'reaper', 'harvester'])) {
      return '$_dir/harvester-Kubota-DC-70.jpg';
    }
    if (_has(n, ['tractor', '4wd', 'mahindra', 'holland', 'john deere'])) {
      return '$_dir/tractor-Mahindra475DI.jpg';
    }

    return '$_dir/tractor-New-Holland3630TX.jpeg';
  }
}
