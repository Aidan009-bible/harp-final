# HarpHand UI design research

This redesign treats HarpHand as a research instrument rather than a generic upload form. The visual system is grounded in the saung's lacquered body, vermilion cords, gold ornament, sweeping silhouette, and visible string field. The interface uses those cues abstractly while keeping the actual instrument photography central.

## Sources and design consequences

### Cultural material language

The Metropolitan Museum of Art describes the Burmese saung-gauk as an arched harp with a gold-lacquered belly, red twisted cotton tuning cords, gold ornament, and a black decorative field. Those materials informed the restrained palette: lacquer black, vermilion, burnished gold, and warm ivory.

Source: <https://www.metmuseum.org/art/collection/search/502040>

### Motion accessibility

WCAG guidance notes that nonessential interaction-triggered animation should be suppressible and identifies `prefers-reduced-motion` as a sufficient technique. HarpHand therefore disables entrance, ambient, hover movement, and signal animation when the operating system requests reduced motion.

Sources:

- <https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions>
- <https://www.w3.org/WAI/WCAG22/Techniques/css/C39>

### Animation performance

Web performance guidance recommends relying primarily on `transform` and `opacity` for smooth animation and avoiding geometry-changing animation where possible. HarpHand's reveal, hover, image, ambient-light, and signal animations follow that rule.

Source: <https://web.dev/articles/animations-and-performance/>

### Expressive but purposeful motion

Material Design 3's expressive-motion guidance frames motion as a system that should feel natural and help users understand transitions. HarpHand uses a small motion vocabulary rather than unrelated effects: rise-and-fade entrances, restrained image scale, signal amplitude, and short directional feedback.

Source: <https://m3.material.io/styles/motion>

## Product decisions

1. The landing page leads with the instrument and research outcome rather than listing features immediately.
2. The sixteen-string motif appears as a signal visualization, giving the project an ownable interface language without pretending to show a live result.
3. The studio uses the same palette and typography as the landing page, but reduces decorative motion around the core upload task.
4. Result language continues to call audio/hand comparison “agreement,” not accuracy.
5. The profile route no longer pretends that a locally stored display name is a registered account. Google authentication remains available only when configured.
6. Keyboard focus, semantic labels, contrast, responsive layouts, and reduced-motion behavior are first-class constraints.
