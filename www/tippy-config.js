/**
 * Tippy.js Configuration for Molecule Structure Tooltips
 * Displays molecule images on hover over SMILES strings
 */

// Initialize Tippy.js for SMILES hover tooltips
function initializeMoleculeTooltips() {
  // Wait for DataTable to render
  setTimeout(() => {
    const smilesElements = document.querySelectorAll('.smiles-hover');
    
    if (smilesElements.length === 0) {
      console.log('No SMILES elements found yet, will retry...');
      return;
    }
    
    console.log(`Initializing tooltips for ${smilesElements.length} SMILES elements`);
    
    smilesElements.forEach(element => {
      // Destroy existing tooltip instance if any
      if (element._tippy) {
        element._tippy.destroy();
        console.log('Destroyed old tooltip instance');
      }
      
      const molId = element.getAttribute('data-mol-id');
      // Add timestamp to force fresh image load from the start
      const timestamp = Date.now();
      const imgSrc = `molecules/mol_${molId}.png?v=${timestamp}`;
      
      tippy(element, {
        content: `<img src="${imgSrc}" alt="Molecule ${molId}" style="max-width: 300px; border-radius: 8px;" />`,
        allowHTML: true,
        theme: 'molecule',
        placement: 'right',
        arrow: true,
        delay: [200, 0],
        duration: [300, 200],
        maxWidth: 350,
        interactive: true,
        appendTo: document.body,
        onShow(instance) {
          // Generate a fresh timestamp on each show to ensure latest image
          const freshTimestamp = Date.now();
          const imgSrcFresh = `molecules/mol_${molId}.png?v=${freshTimestamp}`;
          
          console.log(`Loading molecule image: ${imgSrcFresh}`);
          
          // Check if image exists, if not, show placeholder
          const img = new Image();
          img.onerror = function() {
            console.error(`Failed to load molecule image: ${imgSrcFresh}`);
            instance.setContent(`<div style="padding: 10px; color: #666;">Molecule image not available</div>`);
          };
          img.onload = function() {
            console.log(`Successfully loaded molecule image: ${imgSrcFresh}`);
            // Update tooltip content with fresh image URL
            instance.setContent(`<img src="${imgSrcFresh}" alt="Molecule ${molId}" style="max-width: 300px; border-radius: 8px;" />`);
          };
          img.src = imgSrcFresh;
        }
      });
    });
  }, 500);
}

// Re-initialize tooltips when DataTable redraws (pagination, search, etc.)
if (typeof $ !== 'undefined') {
  $(document).on('draw.dt', function() {
    console.log('DataTable redrawn, reinitializing tooltips...');
    initializeMoleculeTooltips();
  });
}

// Initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeMoleculeTooltips);
} else {
  initializeMoleculeTooltips();
}

// Also watch for new content (Shiny may update the table)
const observer = new MutationObserver((mutations) => {
  const hasNewTable = mutations.some(mutation => 
    Array.from(mutation.addedNodes).some(node => 
      node.nodeType === 1 && (node.matches('.dataTables_wrapper') || node.querySelector('.dataTables_wrapper'))
    )
  );
  
  if (hasNewTable) {
    console.log('New DataTable detected, initializing tooltips...');
    initializeMoleculeTooltips();
  }
});

// Start observing the document for changes
observer.observe(document.body, {
  childList: true,
  subtree: true
});
