// Structured copy for Crop Types chips on the home dashboard.

class CropGuideBlock {
  final String title;
  final String body;

  const CropGuideBlock({required this.title, required this.body});
}

class CropTypeGuide {
  final String cropName;
  final String? scientificName;
  final String overview;
  final List<CropGuideBlock> growingBasics;
  final List<CropGuideBlock> yieldSyncHelps;

  const CropTypeGuide({
    required this.cropName,
    this.scientificName,
    required this.overview,
    required this.growingBasics,
    required this.yieldSyncHelps,
  });
}

/// Keys must match chip labels on the home screen.
const Map<String, CropTypeGuide> kCropTypeGuides = {
  "Rice": CropTypeGuide(
    cropName: "Rice",
    scientificName: "Oryza sativa",
    overview:
        "Rice is Sri Lanka’s staple cereal and is grown mainly in lowland paddy under flooded or rain-fed conditions. Success depends on timely land preparation, water management, balanced fertiliser use (especially N timing), and harvesting aligned with market windows.",
    growingBasics: [
      CropGuideBlock(
        title: "Seasons",
        body:
            "Commonly aligned with Maha and Yala cultivation cycles.",
      ),
      CropGuideBlock(
        title: "Soil & water",
        body:
            "Heavy clay to loamy soils; reliable irrigation or rainfall matters more than for shallow-rooted vegetables.",
      ),
      CropGuideBlock(
        title: "Key stages",
        body:
            "Nursery / transplant or direct seeding → tillering → flowering / grain filling → harvest.",
      ),
    ],
    yieldSyncHelps: [
      CropGuideBlock(
        title: "IoT & soil",
        body:
            "Monitor pH and nutrients to tune basal and top-dressed fertiliser decisions.",
      ),
      CropGuideBlock(
        title: "Fertiliser module",
        body:
            "Match fertiliser type and rate suggestions to soil readings and growth stage.",
      ),
      CropGuideBlock(
        title: "Market",
        body:
            "Track vegetable and grain price signals where relevant to selling timing.",
      ),
      CropGuideBlock(
        title: "Hiring",
        body:
            "Book labour or machinery (tractors, threshers) when transplanting or harvesting peaks.",
      ),
    ],
  ),
  "Beetroot": CropTypeGuide(
    cropName: "Beetroot",
    scientificName: "Beta vulgaris",
    overview:
        "Beetroot is a root vegetable valued for its bulb (taproot). It prefers cool–moderate climates and uniform moisture; uneven watering or poor drainage often hurts root quality and storability.",
    growingBasics: [
      CropGuideBlock(
        title: "Soil",
        body:
            "Loose, well-drained loam; avoid compaction so roots expand evenly.",
      ),
      CropGuideBlock(
        title: "Nutrition",
        body:
            "Balanced NPK emphasis suitable for root development; avoid excess nitrogen that favours leaves over root bulking.",
      ),
      CropGuideBlock(
        title: "Harvest cue",
        body:
            "Lift when roots reach marketable size; prolonged maturity can cause woodiness.",
      ),
    ],
    yieldSyncHelps: [
      CropGuideBlock(
        title: "IoT + soil",
        body:
            "Track moisture-related stress indicators alongside NPK/pH for better fertiliser timing.",
      ),
      CropGuideBlock(
        title: "Crop advisory",
        body:
            "Align crop choice with current soil and seasonal patterns.",
      ),
      CropGuideBlock(
        title: "Market",
        body:
            "Compare forecast/trend signals for selling decisions (especially short shelf-life roots).",
      ),
      CropGuideBlock(
        title: "Hiring",
        body:
            "Arrange labour for thinning, weeding, or harvest clusters during labour shortages.",
      ),
    ],
  ),
  "Radish": CropTypeGuide(
    cropName: "Radish",
    scientificName: "Raphanus sativus",
    overview:
        "Radish is a fast-maturing root crop used for fresh consumption and short rotations. It is sensitive to soil texture, moisture swings, and nutrient imbalance (often visible as splitting or poor root shape).",
    growingBasics: [
      CropGuideBlock(
        title: "Soil",
        body:
            "Fine tilth, good drainage; stones or hard pans distort roots.",
      ),
      CropGuideBlock(
        title: "Nutrition",
        body:
            "Moderate fertility; avoid heavy nitrogen that promotes excessive foliage.",
      ),
      CropGuideBlock(
        title: "Cycle",
        body:
            "Short duration — timing and irrigation discipline matter more than for long-season crops.",
      ),
    ],
    yieldSyncHelps: [
      CropGuideBlock(
        title: "IoT + soil",
        body:
            "Use live readings to avoid drought/overwater swings that cause cracking or bolting risk patterns.",
      ),
      CropGuideBlock(
        title: "Fertiliser module",
        body:
            "Tie recommendations to soil test-style inputs (NPK, pH) and crop stage.",
      ),
      CropGuideBlock(
        title: "Market",
        body:
            "React quickly to weekly price movements typical for short-cycle vegetables.",
      ),
      CropGuideBlock(
        title: "Hiring",
        body:
            "Quickly find seasonal labour or small equipment for bed preparation and harvest waves.",
      ),
    ],
  ),
  "Red Onion": CropTypeGuide(
    cropName: "Red Onion",
    scientificName: "Allium cepa",
    overview:
        "Red onion is a high-value bulb crop with strong sensitivity to season, storage expectations, and volatile markets. Bulb formation and skin quality depend on nutrition balance, water stress management near maturity, and disease pressure control.",
    growingBasics: [
      CropGuideBlock(
        title: "Soil",
        body:
            "Well-drained soils; waterlogging increases disease risk.",
      ),
      CropGuideBlock(
        title: "Nutrition",
        body:
            "Careful nitrogen management through bulbing; potassium often linked to shelf-life and quality perceptions.",
      ),
      CropGuideBlock(
        title: "Risk",
        body:
            "Price volatility is common — farmers benefit from trend-aware selling decisions.",
      ),
    ],
    yieldSyncHelps: [
      CropGuideBlock(
        title: "IoT + soil",
        body:
            "Monitor key soil parameters to refine fertiliser splits across vegetative vs bulbing phases (aligned with your advisory workflow).",
      ),
      CropGuideBlock(
        title: "Market module",
        body:
            "Especially relevant for onion due to volatile prices — use forecasts/trends for sell/hold-style decisions.",
      ),
      CropGuideBlock(
        title: "Fertiliser module",
        body:
            "Reduce guesswork on nutrient programmes tied to soil readings.",
      ),
      CropGuideBlock(
        title: "Hiring",
        body:
            "Coordinate labour for transplanting, weeding, or harvest peaks when labour demand spikes locally.",
      ),
    ],
  ),
};
