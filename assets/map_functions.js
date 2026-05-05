/* Dash Leaflet 1.x requires JS functions defined on window and referenced
   via {"variable": "path.to.func"} — raw string expressions are NOT eval'd. */

window.mapFunctions = {

    /* ── Site point markers (individual + cluster) ─────────────────────── */
    sitePointToLayer: function (feature, latlng, context) {
        var p = feature.properties;

        if (p.cluster) {
            var n = p.point_count || 0;
            var s = n < 10 ? 30 : n < 100 ? 38 : 46;
            return L.marker(latlng, {
                icon: L.divIcon({
                    html:
                        '<div style="' +
                        'background:#64748b;width:' + (s - 4) + 'px;height:' + (s - 4) + 'px;' +
                        'border-radius:50%;border:3px solid #fff;' +
                        'display:flex;align-items:center;justify-content:center;' +
                        'font-size:11px;font-weight:bold;color:#fff">' +
                        n + '</div>',
                    className: '',
                    iconSize:   [s, s],
                    iconAnchor: [s / 2, s / 2]
                })
            });
        }

        var c = p.color || '#94a3b8';
        return L.circleMarker(latlng, {
            radius:      p.selected ? 14 : 10,
            fillColor:   c,
            color:       p.selected ? '#1e293b' : '#fff',
            weight:      p.selected ? 3 : 2,
            fillOpacity: 0.9,
            opacity:     1
        });
    },

    /* ── Site click handler — injects site_key into clickData ───────────
       We stop propagation so DL's plain {latlng,layerPoint,…} handler
       doesn't overwrite our enriched clickData.                          */
    siteOnEachFeature: function (feature, layer, context) {
        if (feature.properties && feature.properties.cluster) return;
        layer.on('click', function (e) {
            L.DomEvent.stopPropagation(e);
            context.setProps({
                clickData: {
                    site_key:  feature.properties.site_key,
                    site_name: feature.properties.site_name,
                    region:    feature.properties.region,
                    latlng:    [e.latlng.lat, e.latlng.lng]
                }
            });
        });
    },

    /* ── Sector cone polygon style ──────────────────────────────────────── */
    sectorStyle: function (feature, context) {
        var p = feature.properties;
        var c = p.color || '#94a3b8';
        return {
            fillColor:   c,
            fillOpacity: p.selected ? 0.55 : 0.25,
            color:       c,
            weight:      p.selected ? 2.5 : 1.5,
            opacity:     0.9
        };
    },

    /* ── Sector click handler — injects cell_name into clickData ─────────*/
    sectorOnEachFeature: function (feature, layer, context) {
        layer.on('click', function (e) {
            L.DomEvent.stopPropagation(e);
            context.setProps({
                clickData: {
                    cell_name: feature.properties.cell_name,
                    site_key:  feature.properties.site_key,
                    latlng:    [e.latlng.lat, e.latlng.lng]
                }
            });
        });
    }
};
